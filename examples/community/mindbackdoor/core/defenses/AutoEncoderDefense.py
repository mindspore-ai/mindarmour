from pathlib import Path
from typing import Union, Dict
import copy
import time

import numpy as np
import mindspore as ms
from mindspore import Tensor
import mindspore.nn as nn

from .base import Base
from ..dataset import DatasetFolder

class AutoEncoder(nn.Cell):
    def __init__(self):
        super().__init__()
        
        self.encoder = nn.SequentialCell([
            nn.Conv2d(3, 12, 4, stride=2, pad_mode='pad', padding=1),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, pad_mode='pad', padding=1), # [batch, 24, 8, 8]
            nn.ReLU(),
            nn.Conv2d(24, 48, 4, stride=2, pad_mode='pad', padding=1), # [batch, 48, 4, 4]
            nn.ReLU(),
            # 可选：更深的压缩
            # nn.Conv2d(48, 96, 4, stride=2, pad_mode='pad', padding=1),
            # nn.ReLU(),
        ])
        
        self.decoder = nn.SequentialCell([
            # nn.Conv2dTranspose(96, 48, 4, stride=2, pad_mode='pad', padding=1),
            # nn.ReLU(),
            nn.Conv2dTranspose(48, 24, 4, stride=2, pad_mode='pad', padding=1),
            nn.ReLU(),
            nn.Conv2dTranspose(24, 12, 4, stride=2, pad_mode='pad', padding=1),
            nn.ReLU(),
            nn.Conv2dTranspose(12, 3, 4, stride=2, pad_mode='pad', padding=1),
            nn.Sigmoid()
        ])

    def construct(self, x: Tensor):
        
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        
        return decoded
    
    
class AutoEncoderDefense(Base):
    """
    AutoEncoder Defense, a preprocess defense method.
    
    Args:
        train_dataset (DatasetFolder): Training dataset.
        test_dataset (DatasetFolder): Testing dataset.
        seed (int): Random seed.
        pretrain_path (str): Path to the pretrained autoencoder checkpoint.
    """
    def __init__(
        self,
        train_dataset: DatasetFolder,
        test_dataset: DatasetFolder,
        seed: int = 66,
        pretrain_path: Union[str, None] = None,
    ):
        super().__init__(seed=seed)
        
        self.autoencoder: nn.Cell = AutoEncoder()
        # if pretrain_path is not None:
        if pretrain_path is not None:
            try:
                param_dict = ms.load_checkpoint(pretrain_path)
                ms.load_param_into_net(self.autoencoder, param_dict)

                print("Load autoencoder from checkpoint.")
                print(type(self.autoencoder))

            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                print("Train the autoencoder from scratch.")
                self.train_autoencoder(
                    train_dataset=train_dataset,
                    test_dataset=test_dataset,
                )
        else:
            print("No pretrain_path specified, train autoencoder from scratch.")
            self.train_autoencoder(
                train_dataset=train_dataset,
                test_dataset=test_dataset,
            )
        
    def get_autoencoder(self):
        return self.autoencoder
    
    def train_autoencoder(
        self,
        train_dataset: DatasetFolder,
        test_dataset: DatasetFolder,
        schedule: Union[Dict, None] = None,
    ):
        
        if schedule is not None:
            current_schedule = copy.deepcopy(schedule)
        else:
            current_schedule = {
                'device': 'GPU',
                'CUDA_SELECTED_DEVICES': '0',
                
                'batch_size': 16,
                'num_workers': 2,

                'lr': 0.001,
                'betas': (0.9, 0.999),
                'eps': 1e-08,
                'weight_decay': 0,
                'amsgrad': False,
                
                'gamma': 0.1,
                'epochs': 100,
                
                'log_iteration_interval': 100,
                'test_epoch_interval': 10,
            }
    
        if current_schedule.get('device', 'CPU') == 'GPU':
            print('==========Use GPUs to train==========')
            
            selected_devices = current_schedule.get('CUDA_SELECTED_DEVICES', None)
            if selected_devices is None:
                raise AttributeError("CUDA_SELECTED_DEVICES is not set, please set it before training.")
            
            selected_devices = sorted(selected_devices.split(','))
            # we just use the first GPU(the smallest index) to train the model
            selected_device = selected_devices[0]
            
            try:
                ms.set_context(device_target='GPU', device_id=int(selected_device))
            except Exception as e:
                print(f"Error setting GPU context: {e}")
                print("Use CPU to train the autoencoder.")
                ms.set_context(device_target='CPU')
            
        else:
            ms.set_context(device_target='CPU')
            
        train_loader = train_dataset.to_generator_dataset(
            batch_size=current_schedule['batch_size'],
            shuffle=True,
            num_parallel_workers=current_schedule['num_workers']
        )
        
        loss_func = nn.BCELoss(reduction='mean')
        optimizer = nn.Adam(
            self.autoencoder.trainable_params(), 
            learning_rate=current_schedule['lr'], 
            beta1=current_schedule['betas'][0],
            beta2=current_schedule['betas'][1],
            eps=current_schedule['eps'], 
            weight_decay=current_schedule['weight_decay'], 
            use_amsgrad=current_schedule['amsgrad']
            )

        iteration = 0
        last_time = time.time()

        msg = "Total train samples: " + str(len(train_dataset)) + "\n"
        msg += "Total test samples: " + str(len(test_dataset)) + "\n"
        msg += "Batch size: " + str(current_schedule['batch_size']) + "\n"
        msg += "iteration every epoch: " + str(len(train_dataset) // current_schedule['batch_size']) + "\n"
        msg += "Initial learning rate: " + str(current_schedule['lr']) + "\n"
        
        print(msg)
        
        net_with_loss = nn.WithLossCell(self.autoencoder, loss_func)
        
        train_step = nn.TrainOneStepCell(net_with_loss, optimizer)
        train_step.set_train()
        
        epoch_list = []
        acc_list = []
        
        for i in range(current_schedule['epochs']):
            for batch_id, (images, labels) in enumerate(train_loader):
                
                images = Tensor(images, dtype=ms.float32)
                # preds = self.autoencoder(images)
                
                loss_value = train_step(images, images)
                
                iteration += 1
                if iteration % current_schedule['log_iteration_interval'] == 0:
                    msg = time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + f"Epoch:{i+1}/{current_schedule['epochs']}, iteration:{iteration // current_schedule['batch_size']}/{len(train_dataset)//current_schedule['batch_size']}, loss: {float(loss_value)}, time: {time.time()-last_time}\n"
                    print(msg)
                    
            # if (i + 1) % current_schedule['test_epoch_interval'] == 0 or (i + 1) == current_schedule['epochs']:
            #     epoch_list.append(i + 1)
                
            #     predict_logits, labels, _ = self._test_autoencoder(
            #         dataset=test_dataset, 
            #         batch_size=current_schedule['batch_size'], 
            #         num_workers=current_schedule['num_workers'],  
            #         test_loss=loss_func)
                
            #     acc = self._accuracy(predict_logits, labels)
            #     acc_list.append(acc)
                
            #     msg = "==========Test result on benign test dataset==========\n" + \
            #           time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
            #           f"Time: {time.time()-last_time}\n" + \
            #           f"ACC: {acc}\n"
                
            #     print(msg)
                
        return epoch_list, acc_list
    
    def preprocess(self, data: Tensor):
        """Perform AutoEncoder defense method on data and return the preprocessed data.

        Args:
            data (Tensor): Input data (between 0.0 and 1.0), shape: (N, C, H, W) or (C, H, W), dtype: ms.float32.

        Returns:
            Tensor: The preprocessed data.
        """
        
        self.autoencoder.set_train(False)
        if data.ndim == 3:
            # (C, H, W) -> (1, C, H, W)
            preprocessed_data = self.autoencoder(data)
            return preprocessed_data[0]  # no batch dimension
        else:
            # (N, C, H, W)
            preprocessed_data = self.autoencoder(data)
            return preprocessed_data
    
    
    def _test(self, dataset, batch_size=128, num_workers=1, model=None, test_loss=None):
        """
        Test tool for autoencoder, simulate the preprocess
        """
        
        data_loader = dataset.to_generator_dataset(
            batch_size=batch_size,
            shuffle=False,
            num_parallel_workers=num_workers
        )
        
        predict_logits = []
        label_list = []
        loss_list = []
        
        print('Type of model in AutoEncoder Defense test')
        print(type(model))

        model.set_train(False)

        for idx, (batch_img, batch_label) in enumerate(data_loader):
            # preprocess the batch_img
            batch_img = self.preprocess(batch_img)
            
            output = model(batch_img)
            
            loss = test_loss(output, batch_label)

            predict_logits.append(output.asnumpy())
            label_list.append(batch_label.asnumpy())
            loss_list.append(np.array([loss.asnumpy()]))

              
        predict_logits = Tensor(np.concatenate(predict_logits, axis=0))
        label_list = Tensor(np.concatenate(label_list, axis=0))
        loss_list = np.concatenate(loss_list, axis=0)
        
        avg_loss = float(loss_list.mean())
        
        model.set_train(True)
        
        return predict_logits, label_list, avg_loss
        

        
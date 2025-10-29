"""
This file is the base script of the backdoor attacks implementation in this framework
"""

from copy import deepcopy
import os
import random
import time
from typing import List, Tuple, Any, Dict, Optional

import numpy as np
from mindspore import Tensor
import mindspore as ms
import mindspore.nn as nn
from mindspore.nn.learning_rate_schedule import WarmUpLR, ExponentialDecayLR, LearningRateSchedule, CosineDecayLR

from ..dataset import DatasetFolder

def _accuracy(preds: Tensor, labels: Tensor) -> float:
    """
    Compute the accuracy of the models
    
    Args:
        preds: The predicted logits of the model, Tensor[float32]
        labels: The true labels of the dataset, Tensor[uint32]
        
    Returns:
        acc: The accuracy of the model, float
    """
    preds = preds.argmax(axis=1)
    correct = preds.eq(labels)
    acc = correct.sum().item() / len(correct)
    
    return acc

class _WarmupStepDecayLR(LearningRateSchedule):
    def __init__(
        self,
        base_lr: float,
        decay_steps: int,
        decay_rate: float,
        warmup_steps: Optional[int] = None,
    ):
        super().__init__()
        if warmup_steps is not None:
            self.warmup = WarmUpLR(base_lr, warmup_steps)
        else:
            self.warmup = None
        self.decay = ExponentialDecayLR(base_lr, decay_rate, decay_steps, is_stair=True)
        self.warmup_steps = warmup_steps

    def construct(self, global_step):
        if self.warmup is not None and global_step < self.warmup_steps:
            return self.warmup(global_step)
        else:
            return self.decay(global_step)

  
class ModifyTarget:
    """Modify the target of the poisoned image.
    
    Args:
        target (int): The target label.
    """
    def __init__(self, y_target: int):
        self.y_target = y_target
    
    def __call__(self, y_target: int):
        """
        Get the poisoned image
        """
        return self.y_target


class Base(object):
    """
    Base class for backdoor training and testing
    
    Args:
        train_dataset: Benign training dataset, Required
        test_dataset: Benign testing dataset, Required
        model (mindspore.nn.Cell): Network, Required
        loss (mindspore.nn.Cell): Loss function, Required
        schedule (dict): Training schedule, default is None
        poisoned_train_dataset: Poisoned training dataset, Optional, default is None
        poisoned_test_dataset: Poisoned testing dataset, Optional, default is None
        seed (int): Random seed, Optional, default is 66
    """
    
    def __init__(
        self,
        train_dataset: DatasetFolder,
        test_dataset: DatasetFolder,
        model: nn.Cell,
        loss: nn.Cell,
        schedule: Optional[Dict[str, Any]] = None,
        poisoned_train_dataset: Optional[DatasetFolder] = None,
        poisoned_test_dataset: Optional[DatasetFolder] = None,
        seed: int = 66,
    ):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.model = model
        self.loss = loss
        self.schedule = schedule
       
        self.global_scheudle = deepcopy(schedule)
        self.current_schedule = None
        
        self.poisoned_train_dataset = poisoned_train_dataset
        self.poisoned_test_dataset = poisoned_test_dataset
        self.seed = seed

    def _set_seed(self):
        """
        Set the seed for the random number generator
        """
        ms.set_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        os.environ["PYTHONHASHSEED"] = str(self.seed)
    
    def get_model(self):
        return self.model
    
    def get_poisoned_dataset(self):
        return self.poisoned_train_dataset, self.poisoned_test_dataset
    
    def compute_asr(self) -> float:
        """
        Compute the ASR of the backdoor attack
        
        Returns:
            asr: The ASR of the model, float
        """
        # after complete the basic func of base
        pass
    
    
    
    def train(self, schedule: Optional[Dict[str, Any]] = None):
        """
        This script is used to train the model in benign setting or poisoned setting
        
        Args:
            schedule: The schedule of the training, default is None
        """
        if schedule is not None:
            self.current_schedule = deepcopy(schedule)
        elif self.global_schedule is not None:
            self.current_schedule = deepcopy(self.global_schedule)
        else:
            raise AttributeError("Training schedule is None, please check your schedule setting.")
    
        print(f'==========Schedule parameters==========')
        print()
        print(str(self.current_schedule))
        print()
        
        # device setting
        if self.current_schedule.get('device', 'CPU') == 'GPU':
            print('==========Use GPUs to train==========')
            
            selected_devices = self.current_schedule.get('CUDA_SELECTED_DEVICES', None)
            if selected_devices is None:
                raise AttributeError("CUDA_SELECTED_DEVICES is not set, please set it before training.")
            
            selected_devices = sorted(selected_devices.split(','))
            # we just use the first GPU(the smallest index) to train the model
            selected_device = selected_devices[0]
            try:
                ms.set_context(device_target='GPU', device_id=int(selected_device))
            except Exception as e:
                print(f"Error setting GPU context: {e}")
                print("Use CPU to train the model.")
                ms.set_context(device_target='CPU')
                
        else:
            ms.set_context(device_target='CPU')
        
        # dataset setting
        if self.current_schedule['benign_training'] is True:
            train_loader = self.train_dataset.to_generator_dataset(
                batch_size=self.current_schedule['batch_size'],
                shuffle=True,
                num_parallel_workers=self.current_schedule['num_workers']
            )
        elif self.current_schedule['benign_training'] is False:
            train_loader = self.poisoned_train_dataset.to_generator_dataset(
                batch_size=self.current_schedule['batch_size'],
                shuffle=True,
                num_parallel_workers=self.current_schedule['num_workers']
            )
        else:
            raise AttributeError("self.current_schedule['benign_training'] should be True or False.")
        
        # model training
        self.model.set_train(True)
        
        # lr_scheduler = _WarmupStepDecayLR(
        #     base_lr=self.current_schedule['lr'],
        #     warmup_steps=self.current_schedule['warmup_epoch'],
        #     decay_steps=self.current_schedule['decay_epoch'],
        #     decay_rate=self.current_schedule['gamma']
        # )
        
        lr_scheduler = CosineDecayLR(
            min_lr=0.0001,
            max_lr=self.current_schedule['lr'],
            decay_steps=self.current_schedule['decay_epoch']
        )
        
        optimizer = nn.SGD(
            params=self.model.trainable_params(),
            learning_rate=lr_scheduler,
            momentum=self.current_schedule['momentum'],
            weight_decay=self.current_schedule['weight_decay']
        )
        
        iteration = 0
        last_time = time.time()
        
        epoch_list = []
        acc_list = []
        asr_list = []
        
        net_with_loss = nn.WithLossCell(self.model, self.loss)

        
        train_step = nn.TrainOneStepCell(net_with_loss, optimizer)
        train_step.set_train()
        
        for i in range(self.current_schedule['epochs']):
            for batch_id, (images, labels) in enumerate(train_loader):
                
                images = Tensor(images, dtype=ms.float32)
                labels = Tensor(labels, dtype=ms.int32)
                
                loss_value = train_step(images, labels)
                
                iteration += 1
                if iteration % self.current_schedule['log_iteration_interval'] == 0:
                    msg = time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + f"Epoch:{i+1}/{self.current_schedule['epochs']}, iteration:{iteration // self.current_schedule['batch_size']}/{len(self.train_dataset)//self.current_schedule['batch_size']}, loss: {float(loss_value)}, time: {time.time()-last_time}\n"
                    print(msg)
                    
                    
            if (i + 1) % self.current_schedule['test_epoch_interval'] == 0 or (i + 1) == self.current_schedule['epochs']:
                epoch_list.append(i + 1)
                
                
                predict_logits, labels, _ = self._test(
                    self.test_dataset, 
                    self.current_schedule['batch_size'], 
                    self.current_schedule['num_workers'],
                )
                
                acc = _accuracy(predict_logits, labels)
                acc_list.append(acc)
                
                msg = "==========Test result on benign test dataset==========\n" + \
                      time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                      f"Time: {time.time()-last_time}\n" + \
                      f"ACC: {acc}\n"
                
                print(msg)
                
                if self.poisoned_test_dataset is not None:
                    all_poisoned_predict_logits, all_poisoned_labels, _ = self._test(
                        self.poisoned_test_dataset,
                        self.current_schedule['batch_size'],
                        self.current_schedule['num_workers']
                    )
                    
                    asr = _accuracy(all_poisoned_predict_logits, all_poisoned_labels)
                    asr_list.append(asr)
                    
                    msg += "==========Test result on poisoned test dataset==========\n" + \
                        time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                        f"Time: {time.time()-last_time}\n" + \
                        f"ASR: {asr}\n"
                
                print(msg)
                
        self.model.set_train(False)
        
        return epoch_list, acc_list, asr_list
    
    def _test(self, dataset, batch_size=128, num_workers=8, model=None, test_loss=None):
        if model is None:
            model = self.model
        else:
            model = model
            
        if test_loss is None:
            test_loss = self.loss
        else:
            test_loss = test_loss
        
        data_loader = dataset.to_generator_dataset(
            batch_size=batch_size,
            shuffle=False,
            num_parallel_workers=num_workers
        )
        
        predict_logits = []
        label_list = []
        loss_list = []
        
        model.set_train(False)
        
        for batch_img, batch_label in data_loader:
            
            output = model(batch_img)
            loss = test_loss(output, batch_label)

            predict_logits.append(output.asnumpy())
            label_list.append(batch_label.asnumpy())
            loss_list.append(np.array([loss.asnumpy()]))
            
        predict_logits = Tensor(np.concatenate(predict_logits, axis=0))
        label_list = Tensor(np.concatenate(label_list, axis=0))
        loss_list = np.concatenate(loss_list, axis=0)
        
        avg_loss = float(loss_list.mean())
        
        self.model.set_train(True)
        
        return predict_logits, label_list, avg_loss
    
if __name__ == "__main__":
    # simulate results
    preds = Tensor([[0.1, 0.2, 0.7], [0.3, 0.4, 0.3]])
    labels = Tensor([2, 2])
    
    print(_accuracy(preds, labels))
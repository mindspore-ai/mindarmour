from typing import Literal, Any, Dict, Union
import copy
import os
import random
import time

import numpy as np
import mindspore as ms
from mindspore import Tensor, nn
from ..dataset import DatasetFolder

class _CellWithHook(nn.Cell):
    """
    Fake class for typing.
    The model must with hook_dict to store the output of each layer.
    For feature extraction purpose.
    """
    def __init__(self):
        pass
    
    def construct(self, x: Tensor):
        pass
    
    def get_feature(self, layer_name: str):
        return self.hook_dict[layer_name]


class Base(object):
    def __init__(self, seed: int = 66) -> None:
        # self.defense_type = defense_type
        self.seed = seed
        
    def _set_seed(self):
        """
        Set the seed for the random number generator
        """
        ms.set_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        os.environ["PYTHONHASHSEED"] = str(self.seed)
        
    # def detect(self, *args, **kwargs) -> Any:
    #     pass
    
    # def mitigate(self, *args, **kwargs) -> Any:
    #     pass
    
    # def __call__(self, *args, **kwargs) -> Any:
    #     if self.defense_type == 'detection':
    #         return self.detect(*args, **kwargs)
    #     elif self.defense_type == 'mitigation':
    #         return self.mitigate(*args, **kwargs)
    #     else:
    #         raise ValueError(f"Invalid defense type: {self.defense_type}")
        
    def _test(self, dataset: DatasetFolder, model: nn.Cell, batch_size: int = 128, num_workers: int = 1, test_loss: Union[nn.Cell, None] = None):
        """
        Test tool for any model and dataset
        """
        if test_loss is None:
            test_loss = nn.CrossEntropyLoss(reduction='mean')
            
        data_loader = dataset.to_generator_dataset(
            batch_size=batch_size,
            shuffle=False,
            num_parallel_workers=num_workers
        )
        
        predict_logits = []
        label_list = []
        loss_list = []

        model.set_train(False)

        for idx, (batch_img, batch_label) in enumerate(data_loader):

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
    
    def _accuracy(self, preds: Tensor, labels: Tensor) -> float:
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
    
    def test(self, model: nn.Cell, dataset: DatasetFolder, schedule: Union[Dict, None] = None):
        """Uniform test API for any model and any dataset.
        Args:
            model (torch.nn.Module): Network.
            dataset (torch.utils.data.Dataset): Dataset.
            schedule (dict): Testing schedule.
        """
        
        if schedule is not None:
            current_schedule = copy.deepcopy(schedule)
        else:
            current_schedule = {
                'device': 'GPU',
                'CUDA_SELECTED_DEVICES': '0',
                
                'batch_size': 32,
                'num_workers': 1,
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
            
        last_time = time.time()

        predict_logits, labels, _ = self._test(
            dataset=dataset, 
            batch_size=current_schedule['batch_size'], 
            num_workers=current_schedule['num_workers'], 
            model=model, 
            test_loss=nn.CrossEntropyLoss(reduction='mean')
        )
        
        acc = self._accuracy(predict_logits, labels)
        
        msg = "==========Test result on benign test dataset==========\n" + \
              time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
              f"Time: {time.time()-last_time}\n" + \
              f"ACC: {acc}\n"
        
        print(msg)
        
        return acc
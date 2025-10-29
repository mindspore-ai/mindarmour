'''
This is the implement of pruning proposed in [1].
[1] Fine-Pruning: Defending Against Backdooring Attacks on Deep Neural Networks. RAID, 2018.
'''
from typing import Union, Dict, List

import numpy as np
import mindspore as ms
from mindspore import Tensor
import mindspore.nn as nn

from .base import Base, _CellWithHook
from ..dataset import DatasetFolder

class MaskedLayer(nn.Cell):
    def __init__(
        self,
        backbone: nn.Cell,
        mask: Tensor
    ):
        super().__init__()
        self.backbone = backbone
        self.mask = mask
        
    def construct(self, x: Tensor):
        return self.backbone(x) * self.mask


class Pruning(Base):
    """Pruning process.
    Args:
        train_dataset (types in support_list): forward dataset.
        test_dataset (types in support_list): testing dataset.
        model (nn.Cell): Network.
        layers(list): The layers to prune
        prune_rate (float): the pruning rate
        schedule (dict): Training or testing schedule. Default: None.
        seed (int): Global seed for random numbers. Default: 0.
    """
    def __init__(
        self,
        train_dataset: DatasetFolder,
        test_dataset: DatasetFolder,
        model: _CellWithHook,
        layer: str,
        prune_rate: float,
        schedule: Union[Dict, None] = None,
        seed: int = 66,
    ):
        super().__init__(seed=seed)
        
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        # check if the model has attr get_feature
        if not hasattr(model, 'get_feature'):
            raise ValueError("The model must have attr get_feature")
        
        self.model = model
        self.layer = layer
        self.prune_rate = prune_rate
        self.schedule = schedule

    def repair(self, schedule: Union[Dict, None] = None):
        """pruning.
        Args:
            schedule (dict): Schedule for testing.
        """
        
        if schedule is None:
            current_schedule = self.schedule
        else:
            current_schedule = schedule
            
        if current_schedule is None:
            current_schedule = {
                'device': 'GPU',
                'CUDA_SELECTED_DEVICES': '0',
                
                'batch_size': 128,
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
            
        model = self.model
        layer_to_prune = self.layer
        train_loader = self.train_dataset.to_generator_dataset(
            batch_size=current_schedule['batch_size'],
            shuffle=True,
            num_parallel_workers=current_schedule['num_workers']
        )
        prune_rate = self.prune_rate
        
        # prune silent activation
        print("======== pruning... ========")
        model.set_train(False)
        
        container = []
        for _, (images, _) in enumerate(train_loader):
            
            images = Tensor(images, dtype=ms.float32)
            _ = model(images)
            feats = model.get_feature(layer_to_prune)
            
            container.append(feats.asnumpy())
            
        container = np.concatenate(container, axis=0)
        activation = np.mean(container, axis=(0, 2, 3))
        seq_sort = np.argsort(activation)
        num_channels = len(activation)
        prunned_channels = int(num_channels * prune_rate)
        mask = np.ones(num_channels)
        for element in seq_sort[:prunned_channels]:
            mask[element] = 0
        if len(container.shape) == 4:
            mask = mask.reshape(1, -1, 1, 1)    

        # to tensor form
        mask = Tensor(mask, dtype=ms.float32)
        
        setattr(
            model, 
            layer_to_prune, 
            MaskedLayer(getattr(model, layer_to_prune), mask),
        )
        
        self.model = model
        
        return self.model
        
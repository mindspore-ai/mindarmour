"""
This is the implement of fine-tuning proposed in [1].
[1] Fine-Pruning: Defending Against Backdooring Attacks on Deep Neural Networks. RAID, 2018.
"""
import copy
from typing import Union, Dict, List
import time

import numpy as np
import mindspore as ms
from mindspore import Tensor
import mindspore.nn as nn

from .base import Base
from ..dataset import DatasetFolder

class FineTuning(Base):
    """FineTuning process.
    Args:
        train_dataset (DatasetFolder): Benign training dataset.
        test_dataset (DatasetFolder): Benign testing dataset.
        model (nn.Cell): Network.
        layer(list): The layers to fintune
        loss (nn.Cell): Loss.
        schedule (dict): Training or testing schedule. Default: None.
        seed (int): Global seed for random numbers. Default: 0.
    """
    def __init__(
        self,
        train_dataset: DatasetFolder,
        test_dataset: DatasetFolder,
        model: nn.Cell,
        layers: List[str],
        loss: nn.Cell,
        schedule: Union[Dict, None] = None,
        seed: int = 66,
    ):
        super().__init__(seed=seed)

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        # deep copy the model
        self.model = copy.deepcopy(model)
        
        self.finetune_layers = list(layers)
        self.loss = loss
        self.schedule = schedule


    def frozen(self):
        """
        Frozen the layers which dont need to fine-tuning.
        """
        if self.finetune_layers is None or self.finetune_layers[0] == "full layers":
            # 全部层都fine-tune
            return self.model.trainable_params()
        else:
            finetune_param_list = []
            for name, param in self.model.parameters_and_names():
                # 假设name格式如 "layer3.0.conv1.weight"
                if any([name.startswith(layer) for layer in self.finetune_layers]):
                    finetune_param_list.append(param)
            
            return finetune_param_list


    def repair(self, schedule: Union[Dict, None] = None):
        """
        Fine-tuning the model.
        Args:
            schedule (dict): Training or testing schedule. Default: None.
        """

        # frozen the layers which dont need to fine-tuning
        finetune_param_list = self.frozen()
        print('-------------Fine Tuning-------------')
        # set the training schedule
        
        if schedule is None:
            current_schedule = self.schedule

        if current_schedule is None:
            current_schedule = {
                'device': 'GPU',
                'CUDA_SELECTED_DEVICES': '0',

                'batch_size': 128,
                'num_workers': 1,

                'lr': 0.001,
                'momentum': 0.9,
                'weight_decay': 5e-4,
                'gamma': 0.1,
                'schedule': [],

                'epochs': 10,
                'log_iteration_interval': 100,
                'save_epoch_interval': 10,
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

        train_loader = self.train_dataset.to_generator_dataset(
            batch_size=current_schedule['batch_size'],
            shuffle=True,
            num_parallel_workers=current_schedule['num_workers']
        )

        model = self.model
        model.set_train(True)

        optimizer = nn.SGD(
            params=finetune_param_list,
            learning_rate=current_schedule['lr'],
            momentum=current_schedule['momentum'],
            weight_decay=current_schedule['weight_decay'],
        )

        net_with_loss = nn.WithLossCell(self.model, self.loss)
        train_step = nn.TrainOneStepCell(net_with_loss, optimizer)
        train_step.set_train()

        iteration = 0
        last_time = time.time()

        for i in range(current_schedule['epochs']):
            for batch_id, (images, labels) in enumerate(train_loader):
                images = Tensor(images, dtype=ms.float32)
                labels = Tensor(labels, dtype=ms.int32)
                loss_value = train_step(images, labels)

                iteration += 1
                if iteration % current_schedule['log_iteration_interval'] == 0:
                    msg = time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                          f"Epoch:{i + 1}/{current_schedule['epochs']}, " \
                          f"iteration:{iteration // current_schedule['batch_size']}/{len(self.train_dataset) // current_schedule['batch_size']}," \
                          f" loss: {float(loss_value)}, time: {time.time() - last_time}\n"
                    print(msg)
                    last_time = time.time()
                
        train_step.set_train(False)

        return self.model

    def get_model(self):
        return self.model


        
# import os
# import time
# import copy
# from typing import Dict, Any, Optional, List

# import mindspore as ms
# import mindspore.nn as nn
# from mindspore import Tensor

# from .base import Base

# class FineTuning(Base):
#     """
#     Fine-tuning defense in MindSpore (based on Fine-Pruning, RAID 2018)
#     """
#     def __init__(self,
#                  train_dataset=None,
#                  test_dataset=None,
#                  model=None,
#                  finetune_layers: Optional[List[str]] = None,  # 支持 "full layers"
#                  loss=None,
#                  schedule=None,
#                  seed=0):
#         super().__init__(seed=seed)
#         self.train_dataset = train_dataset
#         self.test_dataset = test_dataset
#         self.model = model
#         self.finetune_layers = finetune_layers
#         self.loss = loss
#         self.schedule = schedule

#     def _collect_finetune_params(self):
#         """
#         只收集需要fine-tune的参数, 其它参数自动不更新
#         """
#         if self.finetune_layers is None or self.finetune_layers[0] == "full layers":
#             # 全部层都fine-tune
#             return self.model.trainable_params()
#         else:
#             finetune_param_list = []
#             for name, param in self.model.parameters_and_names():
#                 # 假设name格式如 "layer3.0.conv1.weight"
#                 if any([name.startswith(layer) for layer in self.finetune_layers]):
#                     finetune_param_list.append(param)
#                 else:
#                     param.requires_grad = False  # 其实这行可以不写, 只要optimizer不加它即可
#             return finetune_param_list

#     def repair(self, schedule: Optional[Dict[str, Any]] = None):
#         """
#         Fine-tuning repair (类似于PyTorch的repair)
#         """
#         print("--------fine tuning-------")
#         # 获取/合并 schedule
#         if schedule is not None:
#             current_schedule = copy.deepcopy(schedule)
#         elif self.schedule is not None:
#             current_schedule = copy.deepcopy(self.schedule)
#         else:
#             raise AttributeError("Schedule is None, please check your schedule setting.")

#         # 设备设置
#         if current_schedule.get('device', 'CPU') == 'GPU':
#             print('==========Use GPUs to train==========')
#             selected_devices = current_schedule.get('CUDA_SELECTED_DEVICES', None)
#             if selected_devices is None:
#                 raise AttributeError("CUDA_SELECTED_DEVICES is not set, please set it before training.")
#             selected_device = sorted(selected_devices.split(','))[0]
#             try:
#                 ms.set_context(device_target='GPU', device_id=int(selected_device))
#             except Exception as e:
#                 print(f"Error setting GPU context: {e}")
#                 ms.set_context(device_target='CPU')
#         else:
#             ms.set_context(device_target='CPU')

#         train_loader = self.train_dataset.to_generator_dataset(
#             batch_size=current_schedule['batch_size'],
#             shuffle=True,
#             num_parallel_workers=current_schedule['num_workers']
#         )

#         self.model.set_train(True)
#         # 只优化需要finetune的层
#         finetune_params = self._collect_finetune_params()

#         optimizer = nn.SGD(
#             params=finetune_params,
#             learning_rate=current_schedule['lr'],
#             momentum=current_schedule.get('momentum', 0.9),
#             weight_decay=current_schedule.get('weight_decay', 0.)
#         )

#         net_with_loss = nn.WithLossCell(self.model, self.loss)
#         train_step = nn.TrainOneStepCell(net_with_loss, optimizer)
#         train_step.set_train()

#         iteration = 0
#         last_time = time.time()

#         for i in range(current_schedule['epochs']):
#             for batch_id, (images, labels) in enumerate(train_loader):
#                 images = Tensor(images, dtype=ms.float32)
#                 labels = Tensor(labels, dtype=ms.int32)
#                 loss_value = train_step(images, labels)

#                 iteration += 1
#                 if iteration % current_schedule['log_iteration_interval'] == 0:
#                     msg = time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
#                           f"Epoch:{i + 1}/{current_schedule['epochs']}, " \
#                           f"iteration:{iteration // current_schedule['batch_size']}/{len(self.train_dataset) // current_schedule['batch_size']}," \
#                           f" loss: {float(loss_value)}, time: {time.time() - last_time}\n"
#                     print(msg)
#                     last_time = time.time()

#             # 检查点保存功能，参考你的需求添加
#             # if (i + 1) % current_schedule.get('save_epoch_interval', 5) == 0:
#             #     ms.save_checkpoint(self.model, f"ckpt_epoch_{i+1}.ckpt")

#         self.model.set_train(False)

#     def get_model(self):
#         return self.model

#     def test(self, schedule=None):
#         # 复用Base/test等你自己的
#         super().test(schedule=schedule)
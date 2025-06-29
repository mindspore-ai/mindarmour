"""
This is the implement of pre-processing-based backdoor defense with ShrinkPad proposed in [1].

Reference:
[1] Backdoor Attack in the Physical World. ICLR Workshop, 2021.
"""

import os
import copy
import random
from typing import List, Callable, Union, Dict

import numpy as np
from mindspore.dataset.vision import Pad, Resize, ToTensor
import mindspore.nn as nn

from .base import Base
from ..dataset import Compose, DatasetFolder

class RandomChoice:
    """
    Apply a single transformation among a set of transformations.
    """
    def __init__(self, transforms: List[Callable], p=None):
        self.transforms = transforms
        self.p = p

    def __call__(self, *args, **kwargs):
       t = random.choices(self.transforms, weights=self.p)[0]
       return t(*args, **kwargs)

class ToImageForm:
    """
    Transformed into np.ndarray Image form within value [0, 255) with shape (H, W, C),

    Accept input of np.ndarrat Tensor within value [0, 1) with shape (C, H, W).
    """
    def __init__(self) -> None:
        pass

    def __call__(self, img):
        """
        Args:
            img (np.ndarray): Input image with shape (C, H, W) and value in [0, 1)
            
        Returns:
            np.ndarray: Output image with shape (H, W, C) and value in [0, 255)
        """
        # 转换为 (H, W, C) 格式
        img = img.transpose(1, 2, 0)
        # 缩放到 [0, 255)
        img = img * 255
        return img.astype(np.uint8)
        


def RandomPad(width, height, fill_value=0):
    transfroms_set = []
    
    for i in range(width + 1):
        for j in range(height + 1):
            transfroms_set.append(Pad(padding=(i, j, width - i, height - j), fill_value=fill_value))

    return transfroms_set

class ShrinkPad(Base):
    """
    Construct defense datasets with ShrinkPad method.

    Args:

    """
    def __init__(
        self,
        image_size: int,
        pad_size: int,
        seed: int = 66,
    ):
        super().__init__(seed=seed)
        
        self.image_size = image_size
        self.pad_size = pad_size

        self.shrinkpad_transform = self.build_ShrinkPad(image_size, pad_size)

    def build_ShrinkPad(self, image_size: int, pad_size: int):
        """
        Build the ShrinkPad transform.
        """
        return Compose([
            ToImageForm(),
            Resize((image_size - pad_size, image_size - pad_size)),
            RandomChoice(RandomPad(width=pad_size, height=pad_size)),
            ToTensor()
        ])

    def preprocess(self, data: np.ndarray):
        """
        Perform ShrinkPad defense method on data and return the preprocessed data.

        Args:
            data (np.ndarray): Input data.
            
        Returns:
            np.ndarray: The preprocessed data.
        """
        return self.shrinkpad_transform(data)
        
    def test(self, model: nn.Cell, dataset: DatasetFolder, schedule: Union[Dict, None] = None):
        """Test AutoEncoder on dataset.

        Args:
            model (nn.Cell): Network.
            dataset (DatasetFolder): Dataset.
            schedule (dict): Schedule for testing.
        """
        defense_dataset = copy.deepcopy(dataset)
        
        defense_dataset.transform.transforms.append(self.preprocess)
        
        if hasattr(defense_dataset, 'poisoned_transform'):
            defense_dataset.poisoned_transform.transforms.append(self.preprocess)
        if hasattr(defense_dataset, 'poisoned_test_transform'):
            defense_dataset.poisoned_test_transform.transforms.append(self.preprocess)
        

        print('Type of model in ShrinkPad Defense test')
        print(type(model))
        super().test(
            model=model,
            dataset=defense_dataset,
            schedule=schedule
        )
        
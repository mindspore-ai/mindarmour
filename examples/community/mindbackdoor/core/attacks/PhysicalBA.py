"""
This is the implement of BadNets-based physical backdoor attack proposed in [1].

Reference:
[1] Backdoor Attack in the Physical World. ICLR Workshop, 2021.
"""

import copy
import random
import PIL
from PIL import Image
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
from mindspore import Tensor
import mindspore as ms
from mindspore.dataset.vision import ToTensor
import mindspore.ops as ops
import mindspore.nn as nn

from .base import Base
from .Blended import PoisonedDatasetFolder
from ..dataset import DatasetFolder, Compose

class ColorJitter:
    """
    This is a simple color jitter implementation for facilitating the physical attack.
    """
    def __init__(self, brightness=0.0, contrast=0.0, saturation=0.0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def adjust_brightness(self, img, factor):
        return np.clip(img * factor, 0.0, 1.0)

    def adjust_contrast(self, img, factor):
        mean = img.mean(axis=(1, 2), keepdims=True)
        return np.clip((img - mean) * factor + mean, 0.0, 1.0)

    def adjust_saturation(self, img, factor):
        if img.shape[0] != 3:
            return img  # skip if not RGB
        gray = img.mean(axis=0, keepdims=True)  # shape: (1, H, W)
        return np.clip((img - gray) * factor + gray, 0.0, 1.0)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if not isinstance(img, np.ndarray):
            raise TypeError("Expected input as np.ndarray")
        if img.dtype != np.float32:
            raise TypeError("Expected float32 image input")
        if img.ndim != 3 or img.shape[0] not in [1, 3]:
            raise ValueError("Expected image shape (C, H, W) with C=1 or 3")
        if img.min() < 0.0 or img.max() > 1.0:
            raise ValueError("Expected image in range [0.0, 1.0]")

        ops = []
        if self.brightness > 0:
            factor = random.uniform(1 - self.brightness, 1 + self.brightness)
            ops.append(lambda x: self.adjust_brightness(x, factor))
        if self.contrast > 0:
            factor = random.uniform(1 - self.contrast, 1 + self.contrast)
            ops.append(lambda x: self.adjust_contrast(x, factor))
        if self.saturation > 0:
            factor = random.uniform(1 - self.saturation, 1 + self.saturation)
            ops.append(lambda x: self.adjust_saturation(x, factor))

        random.shuffle(ops)
        for op in ops:
            img = op(img)
        return img

class PhysicalPoisonedDatasetFolder(PoisonedDatasetFolder):
    def __init__(
        self,
        benign_dataset: DatasetFolder,
        y_target: int,
        poisoned_rate: float,
        pattern: np.ndarray,
        weight: np.ndarray,
        poisoned_transform_index: int,
        poisoned_target_transform_index: int,
        physical_transformations: Compose,
    ):
        super(PhysicalPoisonedDatasetFolder, self).__init__(
            benign_dataset,
            y_target,
            poisoned_rate,
            pattern,
            weight,
            poisoned_transform_index,
            poisoned_target_transform_index,
        )
        
        if physical_transformations is None:
            raise ValueError("physical_transformations can not be None.")
        else:
            self.physical_transformations = physical_transformations
        
        def __getitem__(self, index: int):
            """
            Args:
                index (int): Index

            Returns:
                tuple: (sample, target) where target is class_index of the target class.
            """
            path, target = self.samples[index]
            sample = self.loader(path)
            if index in self.poisoned_set:
                sample = self.poisoned_transform(sample)
                sample = self.physical_transformations(sample)
                target = self.poisoned_target_transform(target)
            else:
                if self.transform is not None:
                    sample = self.transform(sample)
                    sample = self.physical_transformations(sample)
                if self.target_transform is not None:
                    target = self.target_transform(target)     

            target = np.int32(target)
            
            return sample, target
        
        
class PhysicalBA(Base):
    """
    
    """
    def __init__(
        self,
        train_dataset: DatasetFolder,
        test_dataset: DatasetFolder,
        model: nn.Cell,
        loss: nn.Cell,
        y_target: int,
        poisoned_rate: float,
        pattern: np.ndarray,
        weight: np.ndarray,
        poisoned_transform_train_index: int = 1,
        poisoned_transform_test_index: int = 1,
        poisoned_target_transform_index: int = 0,
        seed: int = 66,
        physical_transformations: Compose = None,
    ):
        
        poisoned_train_dataset = PhysicalPoisonedDatasetFolder(
            train_dataset,
            y_target,
            poisoned_rate,
            pattern,
            weight,
            poisoned_transform_train_index,
            poisoned_target_transform_index,
            physical_transformations
        )
        
        poisoned_test_dataset = PhysicalPoisonedDatasetFolder(
            test_dataset,
            y_target,
            1.0,
            pattern,
            weight,
            poisoned_transform_test_index,
            poisoned_target_transform_index,
            physical_transformations
        )
        
        super(PhysicalBA, self).__init__(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            model=model,
            loss=loss,
            poisoned_train_dataset=poisoned_train_dataset,
            poisoned_test_dataset=poisoned_test_dataset,
            seed=seed
        )
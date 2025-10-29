'''
This is the implement of BATT [1].

Reference:
[1] BATT: Backdoor Attack with Transformation-based Triggers (ICASSP 2023).
'''

import copy
import random
import PIL
from PIL import Image
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
from mindspore import Tensor
import mindspore as ms
from mindspore.dataset.vision import ToTensor, ToPIL, RandomAffine
import mindspore.ops as ops
import mindspore.nn as nn

from .base import Base, ModifyTarget
from ..dataset import DatasetFolder, Compose

def rotate_np_image(img_np: np.ndarray, angle: float) -> np.ndarray:
    if img_np.shape[0] not in [1, 3]:
        raise ValueError(f"Unsupported channel: {img_np.shape[0]}")
    # C, H, W → H, W, C
    img_np = np.transpose(img_np, (1, 2, 0))
    img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
    img_rotated = img_pil.rotate(angle)
    img_np = np.array(img_rotated).astype(np.float32) / 255.0
    # H, W, C → C, H, W
    img_np = np.transpose(img_np, (2, 0, 1))
    return img_np

class PoisonedDatasetFolder(DatasetFolder):
    def __init__(
        self,
        benign_dataset: DatasetFolder,
        y_target: int,
        poisoned_rate: float,
        poisoned_target_transform_index: int
    ):
        super(PoisonedDatasetFolder, self).__init__(
            root=benign_dataset.root,
            transform=benign_dataset.transform,
            target_transform=benign_dataset.target_transform,
            loader=benign_dataset.loader,
            extensions=benign_dataset.extensions,
        )
        
        total_num = len(benign_dataset)
        poisoned_num = int(total_num * poisoned_rate)
        
        tmp_list = list(range(total_num))
        random.shuffle(tmp_list)

        if poisoned_rate >= 1.0:
            self.poisoned_set = copy.deepcopy(tmp_list)
        else:
            self.poisoned_set = copy.deepcopy(tmp_list[:poisoned_num])
        
        if self.transform is None:
            self.poisoned_transform = Compose([])
        else:
            self.poisoned_transform = copy.deepcopy(self.transform)
        
        if self.target_transform is None:
            self.poisoned_target_transform = Compose([])
        else:
            self.poisoned_target_transform = copy.deepcopy(self.target_transform)
        
        self.poisoned_target_transform.transforms.insert(
            poisoned_target_transform_index,
            ModifyTarget(y_target)
        )
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        transform1 = Compose([RandomAffine(degrees=10)])
        # transform2 = Compose([ToTensor])
        # transform3 = Compose([ToPIL()])
        
        if index in self.poisoned_set:
            sample = self.poisoned_transform(sample)
            sample = rotate_np_image(sample, 16)
            target = self.poisoned_target_transform(target)
        else:
            if self.transform is not None:
                sample = self.transform(sample)
                sample = transform1(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)
            
        target = np.int32(target)
        
        return sample, target
    

class BATT(Base):
    """
    Construct poisoned datasets with BadNets method.
    
    Args:
        train_dataset: Benign training dataset
        test_dataset: Benign testing dataset
        model(nn.Cell): Network.
        loss(nn.Cell): Loss.
        y_target(int): N-to-1 attack target label.
        poisoned_rate(float): Ratio of poisoned samples.
    """
    
    def __init__(
        self,
        train_dataset,
        test_dataset,
        model,
        loss,
        y_target,
        poisoned_rate,
        poisoned_target_transform_index: int = 0,
        seed: int = 66,
    ):
        
        poisoned_train_dataset = PoisonedDatasetFolder(
            train_dataset,
            y_target,
            poisoned_rate,
            poisoned_target_transform_index
        )
        
        poisoned_test_dataset = PoisonedDatasetFolder(
            test_dataset,
            y_target,
            1.0,
            poisoned_target_transform_index
        )
        
        super(BATT, self).__init__(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            model=model,
            loss=loss,
            poisoned_train_dataset=poisoned_train_dataset,
            poisoned_test_dataset=poisoned_test_dataset,
            seed=seed
        )
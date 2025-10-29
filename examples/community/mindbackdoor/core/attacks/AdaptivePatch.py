'''
This is the implement of Adaptive Patch [1] in mindspore.

Reference:
[1] Revisiting the Assumption of Latent Separability for Backdoor Defenses. ICLR, 2023.
'''

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

from .base import Base, ModifyTarget
from ..dataset import DatasetFolder, Compose

class AddTrigger:
    def __init__(self):
        pass

    def add_trigger(self, img: np.ndarray):
        """Add watermarked trigger to image.

        Args:
            img (np.ndarray): shape (C, H, W).

        Returns:
            np.array: Poisoned image, shape (C, H, W).
        """
        return (self.weight * img + self.res).astype(np.float32)
    
class AddDatasetFolderTrigger(AddTrigger):
    """Add watermarked trigger to DatasetFolder images.

    Args:
        pattern (np.ndarray): shape (C, H, W) or (H, W).
        weight (np.ndarray): shape (C, H, W) or (H, W).
    """
    
    def __init__(self, pattern: np.ndarray, weight: np.ndarray):
        super(AddDatasetFolderTrigger, self).__init__()
        
        if pattern is None:
            return ValueError("Pattern can not be None.")
        else:
            self.pattern = pattern
            if self.pattern.ndim == 2:
                # expand to 3 channels
                # self.pattern = self.pattern.unsqueeze(0)
                self.pattern = np.expand_dims(self.pattern, axis=0)
        
        if weight is None:
            return ValueError("Weight can not be None.")
        else:
            self.weight = weight
            if self.weight.ndim == 2:
                # self.weight = self.weight.unsqueeze(0)
                self.weight = np.expand_dims(self.weight, axis=0)
        
        self.res = self.pattern * self.weight
        self.weight = 1.0 - self.weight
        
    def __call__(self, img):
        """
        Get the poisoned image
        """

        def add_trigger(img: Tensor):
            if img.ndim == 2:
                img = img.unsqueeze(0)
                img = self.add_trigger(img)
                img = img.squeeze(0)
            else:
                img = self.add_trigger(img)
            return img
        
        if type(img) == PIL.Image.Image:
            # img: np.ndarray = ToTensor()(img)
            # img = add_trigger(img)
            # # 1 x H x W
            # if img.shape[0] == 1:
            #     img = Image.fromarray(img.squeeze(0), mode='L')
            # # 3 x H x W
            # elif img.shape[0] == 3:
            #     img = Image.fromarray(np.transpose(img, (1, 2, 0)))
            # else:
            #     raise ValueError("Invalid image size.")
            # return img
            img: np.ndarray = np.array(img)  # Convert to numpy (H, W, C) or (H, W)
    
            if img.ndim == 2:
                img = np.expand_dims(img, axis=0)  # (1, H, W)
            elif img.ndim == 3 and img.shape[2] == 3:
                img = np.transpose(img, (2, 0, 1))  # (3, H, W)
            else:
                raise ValueError("Unsupported input image shape.")
            
            # Add trigger
            img = add_trigger(img)

            # Convert to uint8 before Image.fromarray
            # but the image of poisoned_img is [0, 255]
            # img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
            img = img.astype(np.uint8)
            
            if img.shape[0] == 1:
                img = Image.fromarray(img.squeeze(0), mode='L')
            elif img.shape[0] == 3:
                img = Image.fromarray(np.transpose(img, (1, 2, 0)))
            else:
                raise ValueError("Invalid image size after trigger.")
            
            return img
        # numpy
        elif type(img) == np.ndarray:
            img = add_trigger(img)
            return img
        
        # Tensor
        elif type(img) == Tensor:
            # covert to numpy
            img = img.asnumpy()
            img = add_trigger(img)
            # covert to tensor
            img = Tensor(img)
            return img
        else:
            raise ValueError("Invalid image type.")
        
    
class PoisonedDatasetFolder(DatasetFolder):
    def __init__(
        self,
        benign_dataset: DatasetFolder,
        y_target: int,
        poisoned_rate: float,
        cover_rate: float,
        patterns: List[Tensor],
        alphas: List[float],
        poisoned_transform_index,
        poisoned_target_transform_index
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
        
        cover_num = int(total_num * cover_rate)
        
        tmp_list = list(range(total_num))
        random.shuffle(tmp_list)
        
        self.poisoned_set = copy.deepcopy(tmp_list[:poisoned_num])

        self.covered_set = copy.deepcopy(tmp_list[poisoned_num:poisoned_num+cover_num])

        if self.transform is None:
            self.poisoned_transform = Compose([])
        else:
            self.poisoned_transform = copy.deepcopy(self.transform)
        self.add_trigger_transforms = []
        for idx, pattern in enumerate(patterns):
            cond1 = ms.Tensor((pattern[0] > 0), dtype=ms.bool_)
            cond2 = ms.Tensor((pattern[1] > 0), dtype=ms.bool_)
            cond3 = ms.Tensor((pattern[2] > 0), dtype=ms.bool_)
        
            mask = ops.LogicalOr()(ops.LogicalOr()(cond1, cond2), cond3).astype(ms.float32)

            # covert to numpy
            mask = mask.asnumpy()
            
            # weight: [3, H, W] × scalar
            weight = mask * alphas[idx]
            self.add_trigger_transforms.append(AddDatasetFolderTrigger(pattern, weight))
            
        # modifying labels
        self.poisoned_transform_index  = poisoned_transform_index
        self.poisoned_rate = poisoned_rate
        
        if self.poisoned_rate >= 1.0: # poison testset
            for add_trigger_transform in self.add_trigger_transforms[:int(len(patterns)/2)]:
                self.poisoned_transform.transforms.insert(self.poisoned_transform_index, add_trigger_transform)
        
        # modify labels
        if self.target_transform is None:
            self.poisoned_target_transform = Compose([])
        else:
            self.poisoned_target_transform = copy.deepcopy(self.target_transform)
        
        self.poisoned_target_transform.transforms.insert(poisoned_target_transform_index, ModifyTarget(y_target))
        
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): The index.

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.poisoned_rate >= 1.0:       # Poison testset
            sample = self.poisoned_transform(sample)
            # Change labels
            target = self.poisoned_target_transform(target)
        else:       # Poison trainset
            if index in self.poisoned_set:
                idx = self.poisoned_set.index(index) % len(self.add_trigger_transforms)
                poisoned_transform = copy.deepcopy(self.poisoned_transform)
                poisoned_transform.transforms.insert(self.poisoned_transform_index, self.add_trigger_transforms[idx])
                sample = poisoned_transform(sample)
                target = self.poisoned_target_transform(target)
            elif index in self.covered_set:
                idx = self.covered_set.index(index) % len(self.add_trigger_transforms)
                poisoned_transform = copy.deepcopy(self.poisoned_transform)
                poisoned_transform.transforms.insert(self.poisoned_transform_index, self.add_trigger_transforms[idx])
                sample = poisoned_transform(sample)
                # Remain labels
                if self.target_transform is not None:
                    target = self.target_transform(target)
            else:
                if self.transform is not None:
                    sample = self.transform(sample)
                if self.target_transform is not None:
                    target = self.target_transform(target)            

        target = np.int32(target)
        
        return sample, target


class AdaptivePatch(Base):
    """
    Construct poisoned datasets with Adaptive Patch attack.
    
    Args:
        train_dataset (DatasetFolder): The training dataset.
        test_dataset (DatasetFolder): The test dataset.
        model (nn.Cell): The model.
        loss (nn.Cell): The loss function.
        y_target (int): The target label.
        poisoned_rate (float): Ratio of poisoned samples.
        cover_rate (float): Ratio of covered samples.
        patterns (List[Tensor]): The patterns.
        alphas (List[float]): The alphas.
    """
    def __init__(
        self,
        train_dataset: DatasetFolder,
        test_dataset: DatasetFolder,
        model: nn.Cell,
        loss: nn.Cell,
        y_target: int,
        poisoned_rate: float,
        cover_rate: float,
        patterns: List[Tensor],
        alphas: List[float],
        poisoned_transform_train_index: int = 1,
        poisoned_transform_test_index: int = 1,
        poisoned_target_transform_index: int = 0,
        seed: int = 66,
    ):
        
        poisoned_train_dataset = PoisonedDatasetFolder(
            train_dataset,
            y_target,
            poisoned_rate,
            cover_rate,
            patterns,
            alphas,
            poisoned_transform_train_index,
            poisoned_target_transform_index
        )
        
        poisoned_test_dataset = PoisonedDatasetFolder(
            test_dataset,
            y_target,
            1.0, # cuz the testset is fully poisoned
            0,
            patterns,
            [1] * len(alphas),
            poisoned_transform_test_index,
            poisoned_target_transform_index
        )
        
        super(AdaptivePatch, self).__init__(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            model=model,
            loss=loss,
            poisoned_train_dataset=poisoned_train_dataset,
            poisoned_test_dataset=poisoned_test_dataset,
            seed=seed
        )
        
        # done, poisoned trainset and testset are ready
        # begin poisoning!
    
        
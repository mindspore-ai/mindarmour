'''
This is the implement of Adaptive Blended [1] in mindspore.

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
    """
    Adpative Blended Trigger
    """
    def __init__(self, pattern, weight, mask):
        
        if pattern is None:
            return ValueError("Pattern can not be None.")
        else:
            self.pattern = pattern
            if self.pattern.ndim == 2:
                # expand to 3 channels
                # self.pattern = self.pattern.unsqueeze(0)
                self.pattern = np.expand_dims(self.pattern, axis=0)
        
        # if weight is None:
        #     return ValueError("Weight can not be None.")
        # else:
        #     self.weight = weight
        #     if self.weight.ndim == 2:
        #         # self.weight = self.weight.unsqueeze(0)
        #         self.weight = np.expand_dims(self.weight, axis=0)
        self.weight = weight
        self.mask = mask       # shape: (H, W)
        
    def add_trigger(self, img: np.ndarray):
        img = (1 - self.weight) * img + self.weight * (self.pattern * (1 - self.mask) + img * self.mask)
        return img.astype(np.float32)
    
    def __call__(self, img: np.ndarray):
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
            img = self.add_trigger(img)

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
            img = self.add_trigger(img)
            return img
        
        # Tensor
        elif type(img) == Tensor:
            # covert to numpy
            img = img.asnumpy()
            img = self.add_trigger(img)
            # covert to tensor
            img = Tensor(img)
            return img
        else:
            raise ValueError("Invalid image type.")
        

def get_trigger_mask(total_pieces: int, masked_pieces: int, img_size: int = 32):
    """
    Generate a trigger mask for the image.
    """
    div_num = int(np.sqrt(total_pieces))
    step = img_size // div_num
    mask = np.ones((img_size, img_size), dtype=np.float32)
    candidate_idx = random.sample(list(range(total_pieces)), k=masked_pieces)
    for i in candidate_idx:
        x = i % div_num  
        y = i // div_num 
        mask[y*step:(y+1)*step, x*step:(x+1)*step] = 0
    
    return mask


class PoisonedDatasetFolder(DatasetFolder):
    def __init__(
        self,
        benign_dataset: DatasetFolder,
        y_target: int,
        poisoned_rate: float,
        pattern: np.ndarray,
        weight_train: float = 0.15,
        weight_test: float = 0.2,
        pieces: int = 16,
        mask_rate: float = 0.5,
        regularization_ratio: float = 0.5,
        poisoned_transform_index: int = 1,
        poisoned_target_transform_index: int = 1,
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
        
        self.poisoned_set = tmp_list[:poisoned_num]
        
        num_regularization = int(poisoned_num * regularization_ratio)
        self.regularization_indices = self.poisoned_set[:num_regularization]
        self.payload_indices = self.poisoned_set[num_regularization:]
        
        self.pattern = pattern
        self.weight_train = weight_train
        self.weight_test = weight_test
        self.pieces = pieces
        self.masked_pieces = int(pieces * mask_rate)

        self.test_mask = np.zeros((32, 32), dtype=np.float32)
        self.y_target = y_target

        self.poisoned_rate = poisoned_rate
        self.poisoned_transform_index = poisoned_transform_index
        self.poisoned_target_transform_index = poisoned_target_transform_index
        
        if self.transform is None:
            # self.poisoned_train_transform = Compose([])
            self.poisoned_test_transform = Compose([])
            
        else:
            # self.poisoned_train_transform = copy.deepcopy(self.transform)
            self.poisoned_test_transform = copy.deepcopy(self.transform)
        
        # self.poisoned_train_transform.transforms.insert(poisoned_transform_index, AddTrigger(self.pattern, self.weight_train, self.train_mask))
        self.poisoned_test_transform.transforms.insert(poisoned_transform_index, AddTrigger(self.pattern, self.weight_test, self.test_mask))
        
        if self.target_transform is None:
            self.poisoned_target_transform = Compose([])
        else:
            self.poisoned_target_transform = copy.deepcopy(self.target_transform)
        
        self.poisoned_target_transform.transforms.insert(poisoned_target_transform_index, ModifyTarget(self.y_target))
        
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path, target = self.samples[index]
        sample = self.loader(path)
        
        if self.poisoned_rate >= 1.0:
            sample = self.poisoned_test_transform(sample)
            target = self.poisoned_target_transform(target)
        else:
            # Regularization keep the original label
            if index in self.regularization_indices or index in self.payload_indices:
                if self.transform is not None:
                    poisoned_train_transform = copy.deepcopy(self.transform)
                else:
                    poisoned_train_transform = Compose([])
                    
                train_mask = get_trigger_mask(self.pieces, self.masked_pieces)
                poisoned_train_transform.transforms.insert(self.poisoned_transform_index, AddTrigger(self.pattern, self.weight_train, train_mask))
                
                sample = poisoned_train_transform(sample)
                
                if index in self.payload_indices:
                    target = self.poisoned_target_transform(target)
                elif self.target_transform is not None:
                    target = self.target_transform(target)
                    
            else:
                if self.transform is not None:
                    sample = self.transform(sample)
                if self.target_transform is not None:
                    target = self.target_transform(target)
        
        target = np.int32(target)
            
        return sample, target
                

class AdaptiveBlend(Base):
    """
    Construct poisoned datasets with Adaptive Blend attack.
    
    Args:
        train_dataset (DatasetFolder): The training dataset.
        test_dataset (DatasetFolder): The test dataset.
        model (nn.Cell): The model.
        loss (nn.Cell): The loss function.
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
        weight_train: float = 0.15,
        weight_test: float = 0.2,
        pieces: int = 16,
        mask_rate: float = 0.5,
        regularization_ratio: float = 0.5,
        poisoned_transform_train_index: int = 1,
        poisoned_transform_test_index: int = 1,
        poisoned_target_transform_index: int = 0,
        seed: int = 66,
    ):

        poisoned_train_dataset = PoisonedDatasetFolder(
            benign_dataset=train_dataset,
            y_target=y_target,
            poisoned_rate=poisoned_rate,
            pattern=pattern,
            weight_train=weight_train,
            pieces=pieces,
            mask_rate=mask_rate,
            regularization_ratio=regularization_ratio,
            poisoned_transform_index=poisoned_transform_train_index,
            poisoned_target_transform_index=poisoned_target_transform_index,
        )
        
        poisoned_test_dataset = PoisonedDatasetFolder(
            benign_dataset=test_dataset,
            y_target=y_target,
            poisoned_rate=1.0,
            pattern=pattern,
            weight_test=weight_test,
            poisoned_transform_index=poisoned_transform_test_index,
            poisoned_target_transform_index=poisoned_target_transform_index,
        )
        
        super(AdaptiveBlend, self).__init__(
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
    
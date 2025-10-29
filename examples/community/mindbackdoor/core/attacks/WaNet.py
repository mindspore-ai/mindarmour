"""
This is the implement of WaNet [1].

Reference:
[1] WaNet - Imperceptible Warping-based Backdoor Attack. ICLR 2021.
""" 

import copy
import random
import PIL
from PIL import Image
from typing import Optional, Tuple, List, Dict, Any, Union

import numpy as np
from scipy.ndimage import zoom, map_coordinates
from mindspore import Tensor
import mindspore as ms
from mindspore.dataset.vision import ToTensor, ToPIL, RandomAffine
import mindspore.ops as ops
import mindspore.nn as nn

from .base import Base, ModifyTarget
from ..dataset import DatasetFolder, Compose

class AddWaNetTrigger:
    """Add WaNet trigger to DatasetFolder images.

    Args:
        identity_grid (np.ndarray): the poisoned pattern shape.
        noise_grid (np.ndarray): the noise pattern.
        noise (bool): turn on noise mode, default is False.
        strength (int or float): The strength of the noise grid. Default is 0.5.
        grid_rescale (int or float): Scale :attr:`grid` to avoid pixel values going out of [-1, 1].
            Default is 1.
        noise_rescale (int or float): Scale the random noise from a uniform distribution on the
            interval [0, 1). Default is 2.
    """
    def __init__(
        self,
        identity_grid: np.ndarray,
        noise_grid: np.ndarray,
        noise: bool = False,
        strength: int = 0.5,
        grid_rescale: int = 1,
        noise_rescale: int = 2,
    ):
        self.identity_grid = identity_grid
        self.noise_grid = noise_grid
        self.noise = noise
        self.strength = strength
        self.grid_rescale = grid_rescale
        self.noise_rescale = noise_rescale
        
        self.h = self.identity_grid.shape[2]
        grid = self.identity_grid + self.strength * self.noise_grid / self.h
        # if there is any function in numpy, like torch.clamp, to avoid the pixel values going out of [-1, 1]
        self.grid = np.clip(grid * self.grid_rescale, -1, 1)
        self.noise_rescale = noise_rescale
        
    def add_trigger(self, img: np.ndarray, noise: bool = False) -> np.ndarray:
        """
        Add WaNet trigger to image.
        
        Args:
            img (np.ndarray): the image to be poisoned. Input shape is (C, H, W).
            noise (bool): turn on noise mode, default is False.
            
        Returns:
            np.ndarray: the poisoned image in shape (C, H, W).
        """
        
        current_sampling_grid = self.grid
        
        if noise:
            ins = np.random.rand(1, self.h, self.h, 2).astype(np.float32) * self.noise_rescale - 1
            grid = self.grid + ins / self.h
            current_sampling_grid = np.clip(grid, -1.0, 1.0)
        
        
        num_channels, h_in, w_in = img.shape
        
        # Output grid dimensions (square grid of size self.h by self.h)
        grid_h_out, grid_w_out = self.h, self.h

        # Extract normalized grid coordinates (x, y) from the sampling grid.
        # PyTorch grid_sample expects (x,y) where x is horizontal, y is vertical.
        # current_sampling_grid is (1, grid_h_out, grid_w_out, 2)
        norm_coords_x = current_sampling_grid[0, :, :, 0]  # Shape (grid_h_out, grid_w_out)
        norm_coords_y = current_sampling_grid[0, :, :, 1]  # Shape (grid_h_out, grid_w_out)

        # Convert normalized coordinates [-1, 1] to pixel coordinates [0, size-1]
        # This matches `align_corners=True` behavior in `grid_sample`.
        # x_pixel = (x_norm + 1) / 2 * (W_in - 1)
        # y_pixel = (y_norm + 1) / 2 * (H_in - 1)
        pixel_coords_x = (norm_coords_x + 1.0) / 2.0 * (w_in - 1)
        pixel_coords_y = (norm_coords_y + 1.0) / 2.0 * (h_in - 1)

        # `map_coordinates` requires coordinates as (D, N_points), where D is number of dimensions.
        # For 2D image, it's (coords_for_dim0, coords_for_dim1), i.e., (row_coords, col_coords).
        # So, map_coords should be [pixel_coords_y.ravel(), pixel_coords_x.ravel()].
        map_coords = np.array([pixel_coords_y.ravel(), pixel_coords_x.ravel()])

        poisoned_channels_list = []
        for i in range(num_channels):
            input_channel_data = img[i, :, :]
            
            # Perform resampling for the current channel.
            # `order=1` for bilinear interpolation.
            # `mode='constant', cval=0.0` mimics `grid_sample`'s default `padding_mode='zeros'`.
            resampled_channel_flat = map_coordinates(
                input_channel_data,
                map_coords,
                order=1,          # Bilinear interpolation
                mode='constant',  # Padding with a constant value
                cval=0.0,         # Constant value is 0
                prefilter=False   # Not needed for order=0 or 1
            )
            # Reshape the flattened resampled channel to the output grid dimensions
            poisoned_channels_list.append(resampled_channel_flat.reshape(grid_h_out, grid_w_out))
        
        poisoned_img = np.array(poisoned_channels_list) # Shape (C, grid_h_out, grid_w_out)
        # clip to tensor value range
        poisoned_img = np.clip(poisoned_img, 0, 1)
        
        return poisoned_img
    
    def __call__(self, img: Optional[Union[np.ndarray, PIL.Image.Image]]) -> Optional[Union[np.ndarray, PIL.Image.Image]]:
        """
        Add WaNet trigger to image.
        
        Args:
            img (np.ndarray | PIL.Image.Image): the image to be poisoned. Input shape is (H, W, C).
            
        Returns:
            np.ndarray | PIL.Image.Image: the poisoned image in shape (H, W, C).
        """
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
            img: np.ndarray = ToTensor()(img)  # transform to (C, H, W) with value range [0, 1]
            
            # Add trigger
            img = self.add_trigger(img, noise=self.noise)

            # Convert to uint8 before Image.fromarray
            img = np.clip(img * 255.0, 0, 255).astype(np.uint8)

            # return to (H, W, C)
            if img.shape[0] == 1:
                img = Image.fromarray(img.squeeze(0), mode='L')
            elif img.shape[0] == 3:
                img = Image.fromarray(np.transpose(img, (1, 2, 0)))
            else:
                raise ValueError("Invalid image size after trigger.")
            
            return img
        
        elif type(img) == np.ndarray:
            # img has gone through ToTensor()
            img = self.add_trigger(img, noise=self.noise)
            
            # clip to tensor value range
            img = np.clip(img, 0, 1)
            
            return img
        
        else:
            raise ValueError("Unsupported input image type.")
    

class PoisonedDatasetFolder(DatasetFolder):
    def __init__(
        self,
        benign_dataset: DatasetFolder,
        y_target: int,
        poisoned_rate: float,
        identity_grid: np.ndarray,
        noise_grid: np.ndarray,
        noise: bool,
        poisoned_transform_index: int,
        poisoned_target_transform_index: int,
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
        
        self.noise = noise
        
        if poisoned_rate < 1.0:
            noise_rate = poisoned_rate * 2
            noise_num = int(total_num * noise_rate)
            
            
            self.noise_set = tmp_list[poisoned_num: poisoned_num + noise_num]
        else:
            self.noise_set = []
        
        if self.transform is None:
            self.poisoned_transform = Compose([])
            self.poisoned_transform_noise = Compose([]) # add noise
        else:
            self.poisoned_transform = copy.deepcopy(self.transform)
            self.poisoned_transform_noise = copy.deepcopy(self.transform) # add noise    
        
        self.poisoned_transform.transforms.insert(poisoned_transform_index, AddWaNetTrigger(identity_grid, noise_grid,  noise=False))
        #add noise transform
        self.poisoned_transform_noise.transforms.insert(poisoned_transform_index, AddWaNetTrigger(identity_grid, noise_grid,  noise=True))

        # modify labels
        if self.target_transform is None:
            self.poisoned_target_transform = Compose([])
        else:
            self.poisoned_target_transform = copy.deepcopy(self.target_transform)
        self.poisoned_target_transform.transforms.insert(poisoned_target_transform_index, ModifyTarget(y_target))
        
    def __getitem__(self, index):
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
            target = self.poisoned_target_transform(target)
        # add noise mode
        elif index in self.noise_set and self.noise == True:
            sample = self.poisoned_transform_noise(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)
            # target = self.poisoned_target_transform(target)

        else:
            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)

        target = np.int32(target)
        
        return sample, target
    
class WaNet(Base):
    """
    Construct poisoned datasets with WaNet method.

    Args:
        train_dataset: Benign training dataset
        test_dataset: Benign testing dataset
        model(nn.Cell): Network.
        loss(nn.Cell): Loss.
        y_target(int): N-to-1 attack target label.
    """
    def __init__(
        self,
        train_dataset: DatasetFolder,
        test_dataset: DatasetFolder,
        model: nn.Cell,
        loss: nn.Cell,
        y_target: int,
        poisoned_rate: float,
        identity_grid: np.ndarray,
        noise_grid: np.ndarray,
        noise: bool,
        poisoned_transform_train_index: int = 1,
        poisoned_transform_test_index: int = 1,
        poisoned_target_transform_index: int =0,
        seed: int = 66,
    ):
        
        poisoned_train_dataset = PoisonedDatasetFolder(
            train_dataset,
            y_target,
            poisoned_rate,
            identity_grid,
            noise_grid,
            noise,
            poisoned_transform_train_index,
            poisoned_target_transform_index,
        )
        
        poisoned_test_dataset = PoisonedDatasetFolder(
            test_dataset,
            y_target,
            1.0,
            identity_grid,
            noise_grid,
            noise,
            poisoned_transform_test_index,
            poisoned_target_transform_index,
        )
        
        super(WaNet, self).__init__(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            model=model,
            loss=loss,
            poisoned_train_dataset=poisoned_train_dataset,
            poisoned_test_dataset=poisoned_test_dataset,
            seed=seed
        )
    
    @classmethod
    def get_grid_numpy(self, height: int, k: int):
        """
        Generate an identity grid with shape 1*height*height*2 and a noise grid with shape 1*height*height*2
        according to the input height ``height`` and the uniform grid size ``k``.
        """
        
        ins = np.random.rand(1, 2, k, k) * 2 - 1 # (1, 2 ,k ,k)
        ins = ins / np.mean(np.abs(ins))
        
        ins_2k = ins[0] # (2, k, k)
        upsampled = np.stack([
            zoom(ins_2k[0], zoom=height / k, order=3),  # order=3: bicubic
            zoom(ins_2k[1], zoom=height / k, order=3)
        ], axis=0)  # (2, height, height)
        noise_grid = upsampled.transpose((1, 2, 0))[None, ...] # (1, height, height, 2)
        
        array1d = np.linspace(-1, 1, height)
        x, y = np.meshgrid(array1d, array1d)
        identity_grid = np.stack((y, x), axis=2)[None, ...]
        
        return identity_grid.astype(np.float32), noise_grid.astype(np.float32)
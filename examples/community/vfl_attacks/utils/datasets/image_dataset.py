# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
Define image dataset, only support two parties, used for CIFAR and CINIC
"""
from typing import Any, Callable, Optional
import numpy as np
import mindspore as ms

class ImageDataset(object):
    """
    The vfl dataset, support image 3,32,32 and two parties.
    """
    def __init__(
            self,
            X, y,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            backdoor_indices=None,
            half=None,
            trigger=None,
            trigger_add=False,
            source_indices=None
    ) -> None:
        """
        Construction of dataset for scenarios with two participants.

        Args:
            X (ndarray): The data.
            y (ndarray): The labels.
            transform (callable, optional): Transform for image.
            target_transform (callable, optional): Transformer for labels.
            backdoor_indices (ndarray): The indices of poison samples.
            half (int, optional): Default is 16.
            trigger (str, optional): Default is 'pixel'. Add trigger on poison samples; otherwise,
                                do not add trigger on poison samples.
            trigger_add (bool, optional): Whether the trigger is additive or a mask trigger.
            source_indices (list, optional): The indices of samples used to replace poisoned samples.
        """
        self.transform = transform
        self.target_transform = target_transform

        self.data: Any = []
        self.targets = []

        self.data = X
        self.targets = y

        self.backdoor_indices = backdoor_indices
        self.source_indices = source_indices

        if backdoor_indices is not None and source_indices is not None:
            self.indice_map = dict(zip(backdoor_indices, source_indices))
        else:
            self.indice_map = None

        self.half = half
        self.trigger = trigger
        if self.trigger is None:
            self.trigger = 'pixel'
        self.trigger_add = trigger_add
        channels, height, width = self.data[0].shape
        if channels in {1, 3, 4}:
            self.pixel_pattern = np.full((channels, height, half), 0)
        else:
            self.pixel_pattern = np.full((width, height, half), 0)
        
        pattern_mask: ms.Tensor = ms.tensor([
            [1., 0., 1.],
            [-10., 1., -10.],
            [-10., -10., 0.],
            [-10., 1., -10.],
            [1., 0., 1.]
        ], dtype=ms.float32)

        pattern_mask = pattern_mask.unsqueeze(0)
        self.pattern_mask = pattern_mask.tile((3, 1, 1))
        self.pattern_mask = self.pattern_mask.asnumpy()
        x_top = 3
        y_top = 3
        x_bot = x_top + self.pattern_mask.shape[1]
        y_bot = y_top + self.pattern_mask.shape[2]
        self.location = [x_top, x_bot, y_top, y_bot]

    def __getitem__(self, index):
        """
        Get data.

        Args:
            index (int): The index of the sample.

        Returns:
            tuple: A tuple containing:
                imgs (ndarray): Clean data or poisoned data.
                target (int): The label.
                old_img (ndarray): The original data without poisoning.
        """
        index = int(index)
        img, target = self.data[index], self.targets[index]
        if img.shape[0] in {1, 3, 4}:
            img = img.transpose(1, 2, 0)
        if self.transform is not None:
            img = self.transform(img)
        img_a, img_b = img[:, :, :self.half], img[:, :, self.half:]

        if self.target_transform is not None:
            target = self.target_transform(target)

        old_imgb = img_b

        if self.trigger == 'pixel':
            if self.indice_map is not None and index in self.indice_map.keys():
                source_indice = self.indice_map[index]
                source_img = self.data[source_indice]
                source_img = source_img.transpose(1, 2, 0)
                if self.transform is not None:
                    source_img = self.transform(source_img)
                img_b = source_img[:, :, self.half:]

        # add trigger if index is in backdoor indices
        if self.trigger == 'pixel':
            if self.backdoor_indices is not None and index in self.backdoor_indices:
                if self.trigger_add:
                    img_b = img_b + self.pixel_pattern
                else:
                    img_b = _add_pixel_pattern_backdoor(img_b, self.pattern_mask, self.location)
        imgs = (img_a, img_b)
        return imgs, target, old_imgb

    def __len__(self) -> int:
        """
        Get the length of the dataset.
        """
        return len(self.data)


def _add_pixel_pattern_backdoor(inputs, pattern_tensor, location):
    """
    Add pixel pattern trigger to image.

    Args:
        inputs (Tensor): Normal images.
        pattern_tensor (ndarray): The additive trigger.
        location (list or tuple): The area to put the trigger.

    Returns:
        Tensor: Images with the trigger.
    """
    mask_value = -10

    input_shape = inputs.shape
    full_image = np.full(input_shape, mask_value, dtype=inputs.dtype)

    x_top = location[0]
    x_bot = location[1]
    y_top = location[2]
    y_bot = location[3]
    full_image[:, x_top:x_bot, y_top:y_bot] = pattern_tensor

    mask = 1 * (full_image != mask_value)
    pattern = full_image

    inputs = (1 - mask) * inputs + mask * pattern
    inputs = inputs.astype(pattern_tensor.dtype)
    return inputs

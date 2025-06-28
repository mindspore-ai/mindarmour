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
import random
from typing import Any, Callable, Optional, Tuple
import os
import numpy as np
from PIL import Image
import mindspore as ms
from MindsporeCode.datasets.common import add_pixel_pattern_backdoor
from typing import Any, Callable, List, Optional, Tuple

"""
define image dataset, only support two parties, used for CIFAR and CINIC
"""

class ImageDataset(object):

    def __init__(
            self,
            X, y,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            backdoor_indices=None,  # target for ar_ba
            half=None,
            trigger=None,
            trigger_add=False,
            source_indices=None,  # none_target for sr_ba
            root=""
    ) -> None:
        if isinstance(root, (str, bytes)):
            root = os.path.expanduser(root)
        self.root = root

        # for backwards-compatibility
        self.transform = transform
        self.target_transform = target_transform

        self.data: Any = []  # X
        self.targets = []  # Y

        self.data = X
        self.targets = y

        self.backdoor_indices = backdoor_indices  # backdoor indices of dataset
        self.source_indices = source_indices

        if backdoor_indices is not None and source_indices is not None:
            self.indice_map = dict(zip(backdoor_indices, source_indices))
        else:
            self.indice_map = None

        self.half = half  # vertical halves to split
        self.trigger = trigger
        if self.trigger is None:
            self.trigger = 'pixel'
        self.trigger_add = trigger_add
        self.pixel_pattern = np.full((3, 32, 16), 0)

        # pattern_mask: ms.Tensor = ms.tensor([
        #     [1., 0., 1.],
        #     [-10., 1., -10.],
        #     [-10., -10., 0.],
        #     [-10., 1., -10.],
        #     [1., 0., 1.]
        # ])
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
        index = int(index)
        img, target = self.data[index], self.targets[index]  # 3,32,32
        # print('img shape: ', img.shape)
        # img = img.transpose(1, 2, 0)

        # if type(img) is np.str_:
        #     img = Image.open(img)
        # else:
        #     img = Image.fromarray(np.uint8(img.transpose(1, 2, 0)))
        img = img.transpose(1, 2, 0)
        # print('img shape: ', img.shape)
        # print('img type: ', type(img))

        if self.transform is not None:
            img = self.transform(img)
            # print('data type: ', img.dtype)  float 32
        # split image into halves vertically for parties
        img_a, img_b = img[:, :, :self.half], img[:, :, self.half:]  # [3, 32, 16]

        if self.target_transform is not None:
            target = self.target_transform(target)

        old_imgb = img_b

        if self.trigger == 'pixel':
            if self.indice_map is not None and index in self.indice_map.keys():
                source_indice = self.indice_map[index]
                # source_indice = random.sample(self.source_indices, 1)[0]
                source_img = self.data[source_indice]
                # if type(img) is np.str_:
                #     source_img = Image.open(source_img)
                # else:
                #     source_img = Image.fromarray(source_img)
                source_img = source_img.transpose(1, 2, 0)
                if self.transform is not None:
                    source_img = self.transform(source_img)
                img_b = source_img[:, :, self.half:]
                # old_imgb = img_b

        # add trigger if index is in backdoor indices
        if self.trigger == 'pixel':
            if self.backdoor_indices is not None and index in self.backdoor_indices:
                if self.trigger_add:
                    img_b = img_b + self.pixel_pattern
                else:
                    img_b = add_pixel_pattern_backdoor(img_b, self.pattern_mask, self.location)

        # img_a = Image.fromarray(np.uint8(img_a.transpose(1, 2, 0) *255))
        # img_b = Image.fromarray(np.uint8(img_b.transpose(1, 2, 0) *255))

        # if self.transform is not None:
        #     img_a = self.transform(img_a)[0]
        #     img_b = self.transform(img_b)[0]

        # 0-1
        return (img_a, img_b), target, old_imgb


    def __len__(self) -> int:
        return len(self.data)

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
from typing import Any, Callable, Optional, Tuple
import mindspore.dataset as ds
import mindspore as ms
import os
from MindsporeCode.common.utils import FeatureTuple
"""
define image dataset, only support two parties, used for CRITEO
"""

def add_input_pattern_backdoor(data, trigger, location):
    data[location[0]:location[1]] = trigger
    return data


class TabularDataset(object):

    def __init__(
            self,
            X, y,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            backdoor_indices=None,  # target for ar_ba
            half=None,
            trigger=None,
            trigger_add=False,
            root="",
            source_indices=None  # none_target for sr_ba
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

        self.pattern_mask = [1., 0., 1., 0.] * 25
        left = 10
        right = 110
        self.location = [left, right]
        self.pixel_pattern = ms.ops.zeros((3,32,16))

    def __getitem__(self, index: int) -> Tuple[Tuple[Any, Any], Any, Any]:
        index = int(index) #0627添加
        img, target = self.data[index], self.targets[index]

        # split image into halves vertically for parties
        img_a, img_b = img[:self.half], img[self.half:]

        old_imgb = img_b

        if self.trigger == 'pixel':
            if self.indice_map is not None and index in self.indice_map.keys():
                source_indice = self.indice_map[index]
                source_img = self.data[source_indice]
                img_b = source_img[:, :, self.half:]

        # add trigger if index is in backdoor indices
        if self.trigger == 'pixel':
            if self.backdoor_indices is not None and index in self.backdoor_indices:
                if self.trigger_add:
                    img_b = img_b + self.pixel_pattern
                else:
                    img_b = add_input_pattern_backdoor(img_b, self.pattern_mask, self.location)

        return (img_a, img_b), target, old_imgb
        # return {'party_data': [img_a, img_b]}, target, old_imgb
        # return img_a, img_b, target, old_imgb
        # return FeatureTuple(img_a, img_b), target, old_imgb

    def __len__(self) -> int:
        return len(self.data)

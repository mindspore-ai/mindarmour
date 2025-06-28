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

"""
define multiple image dataset, support multiple parties, used for Criteo
"""

"""
VisionDataset类的继承这里换为ds继承，具体的实现方式有些变化，感觉可能会有问题
0625 改成object
"""


def add_input_pattern_backdoor(data, trigger, location):
    data[location[0]:location[1]] = trigger
    return data


class MultiTabularDataset(object):
    def __init__(
            self,
            X, y,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            backdoor_indices=None,
            party_num=2,
            trigger=None,
            trigger_add=False,
            root="",
            source_indices=None,  # none_target
            adversary=1
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
        self.party_num = party_num  # parties number
        self.source_indices = source_indices

        self.pattern_mask = [1., 0., 1., 0.] * 25
        left = 10
        right = 110
        self.location = [left, right]

        if backdoor_indices is not None and source_indices is not None:
            self.indice_map = dict(zip(backdoor_indices, source_indices))
        else:
            self.indice_map = None

        self.trigger = trigger
        if self.trigger is None:
            self.trigger = 'pixel'
        self.trigger_add = trigger_add
        self.attacker = adversary
        self.pixel_pattern = None

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        img_groups, target = self.data[index], self.targets[index]
        if len(img_groups) == self.party_num:
            images_list = []
            old_image = None
            # split images into parties
            for img_id in range(self.party_num):
                img_path = img_groups[img_id]
                image = img_path

                if img_id == self.attacker:
                    old_image = image
                    # sample replace
                    if self.trigger == 'pixel':
                        if self.indice_map is not None and index in self.indice_map.keys():
                            source_indice = self.indice_map[index]
                            source_img_groups = self.data[source_indice]
                            source_img_path = source_img_groups[self.attacker]
                            image = source_img_path
                    # add pixel trigger
                    if self.trigger == 'pixel':
                        if self.backdoor_indices is not None and index in self.backdoor_indices:
                            if not self.trigger_add:
                                image = add_input_pattern_backdoor(image, trigger=self.pattern_mask,
                                                                   location=self.location)
                            else:
                                raise ValueError('not support additive trigger!')
                images_list.append(image)
        else:
            img, target = self.data[index], self.targets[index]

            images_list = self.split_func(img)

            old_image = images_list[self.attacker]
            # sample replace
            if self.trigger == 'pixel':
                if self.indice_map is not None and index in self.indice_map.keys():
                    source_indice = self.indice_map[index]
                    source_img = self.data[source_indice]
                    source_img_list = self.split_func(source_img)
                    source_image = source_img_list[self.attacker]
                    images_list[self.attacker] = source_image

            if self.trigger == 'pixel':
                if self.backdoor_indices is not None and index in self.backdoor_indices:
                    if self.trigger_add:
                        if self.pixel_pattern is None:
                            self.pixel_pattern = ms.ops.full_like(images_list[self.attacker], 0)
                        images_list[self.attacker] = images_list[self.attacker] + self.pixel_pattern
                    else:
                        images_list[self.attacker] = add_input_pattern_backdoor(images_list[self.attacker],
                                                                                self.pattern_mask, self.location)

        if self.party_num < 3:
            images = tuple(image for image in images_list)  # 3,3,50,50
        else:
            images = ms.ops.stack(tuple(image for image in images_list), 0)  # 3,3,50,50
        return images, target, old_image

    def __len__(self) -> int:
        return len(self.data)

    def split_func(self, img):
        images_list = []
        length = 2 ** 13 // self.party_num
        for i in range(self.party_num):
            images_list.append(img[:, :, i * length:(i + 1) * length])
        return images_list

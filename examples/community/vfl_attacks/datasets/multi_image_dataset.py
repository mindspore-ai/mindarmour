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
import os
import mindspore as ms
from MindsporeCode.datasets.common import add_pixel_pattern_backdoor, add_pixel_pattern_backdoor_original
import numpy as np
from PIL import Image


class MultiImageDataset(object):
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
            source_indices=None,  # none_target for sr_ba
            adversary=1
    ) -> None:
        #   ????
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

        pattern_mask: ms.Tensor = ms.tensor([
            [1., 0., 1.],
            [-10., 1., -10.],
            [-10., -10., 0.],
            [-10., 1., -10.],
            [1., 0., 1.]
        ], dtype=ms.float32)    ###
        pattern_mask = pattern_mask.unsqueeze(0)
        self.pattern_mask = pattern_mask.tile((3, 1, 1))
        self.pattern_mask = self.pattern_mask.asnumpy()
        x_top = 3
        y_top = 3
        x_bot = x_top + self.pattern_mask.shape[1]
        y_bot = y_top + self.pattern_mask.shape[2]
        self.location = [x_top, x_bot, y_top, y_bot]

        if backdoor_indices is not None and source_indices is not None:
            self.indice_map = dict(zip(backdoor_indices, source_indices))
        else:
            self.indice_map = None

        self.trigger = trigger
        if self.trigger is None:
            self.trigger = 'pixel'
        self.trigger_add = trigger_add
        self.pixel_pattern = None
        self.attacker = adversary

    def __getitem__(self, index):
        index = int(index)
        img_groups, target = self.data[index], self.targets[index]  # img_groups[N_party,50,50,3]
        # img_groups = img_groups.transpose(1,2,3,0)            #### ????? img_groups[50,50,3,N_party]
        # print(img_groups.shape)
        if len(img_groups) == self.party_num:
            images_list = []
            old_image = 0
            # split images into parties
            for img_id in range(self.party_num):
                img_path = img_groups[img_id]
                image = img_path
                # image = image.transpose(2,0,1)
                # print("before",image.shape)
                if self.transform is not None:
                    image = self.transform(image)
                image = image.transpose(2,0,1)    ## ������ 3��50��50
                # print("after",len(image),len(image[0]),len(image[0][0]))
                # if self.target_transform is not None:
                #     target = self.target_transform(target)

                if img_id == self.attacker:
                    old_image = image    # 50,50,3
                    # sample replace
                    if self.trigger == 'pixel':
                        # print("yes")
                        if self.indice_map is not None and index in self.indice_map.keys():
                            # print("yes1")
                            source_indice = self.indice_map[index]
                            source_img_groups = self.data[source_indice]
                            source_img_path = source_img_groups[self.attacker]
                            source_image = source_img_path
                            # print(source_img_groups)
                            # if type(image) is np.str_:
                            #     source_image = Image.open(source_image)
                            # else:
                            #     source_image = Image.fromarray(np.uint8(source_image))
                            # source_img_groups = source_img_groups.transpose(1, 2, 3, 0)        ###????
                            
                            if self.transform is not None:
                                source_image = self.transform(source_image)[0]  # 50,50,3
                                source_image = source_image.transpose(2, 0, 1)
                                
                            image = source_image
                    # add pixel trigger
                    if self.trigger == 'pixel':
                        if self.backdoor_indices is not None and index in self.backdoor_indices:
                            # print("yes2")
                            if not self.trigger_add:
                                image = add_pixel_pattern_backdoor(image, self.pattern_mask,self.location)
                            else:
                                # raise ValueError('not support additive trigger!')
                                if self.pixel_pattern is None:
                                    self.pixel_pattern = ms.ops.full_like(image,0)
                                image = image + self.pixel_pattern
                images_list.append(image)
        else:
            # TODO fix it
            # 3,32,32 for CIFAR or CINIC
            img, target = self.data[index], self.targets[index]
            if type(img) is np.str_:
                img = Image.open(img)
            else:
                # print(img.shape)
                img = Image.fromarray(np.uint8(img.permute(1, 2, 0) *255))
            if self.transform is not None:
                img = self.transform(img)[0]

            images_list = self.split_func(img)

            old_image = images_list[self.attacker]
            # sample replace
            if self.trigger == 'pixel':
                if self.indice_map is not None and index in self.indice_map.keys():
                    source_indice = self.indice_map[index]
                    source_img = self.data[source_indice]
                    if type(img) is np.str_:
                        source_img = Image.open(source_img)
                    else:
                        source_img = Image.fromarray(np.uint8(source_img.transpose(1, 2, 0) * 255))
                    if self.transform is not None:
                        source_img = self.transform(source_img)[0]
                    source_img_list = self.split_func(source_img)
                    source_image = source_img_list[self.attacker]
                    images_list[self.attacker] = source_image
                    
            # sample replace
            if self.trigger == 'pixel':
                if self.indice_map is not None and index in self.indice_map.keys():
                    source_indice = self.indice_map[index]
                    source_img_groups = self.data[source_indice]
                    source_img_path = source_img_groups[self.attacker]
                    source_image = source_img_path
                            
                    if type(image) is np.str_:
                        source_image = Image.open(source_image)
                    else:
                        source_image = Image.fromarray(np.uint8(source_image.transpose(1, 2, 0) * 255))     ### 
                    # source_img_groups = source_img_groups.transpose(1, 2, 3, 0)        ###????
                            
                    if self.transform is not None:
                        source_image = self.transform(source_image)
                                
                    image = source_image

            # add pixel trigger
            if self.trigger == 'pixel':
                if self.backdoor_indices is not None and index in self.backdoor_indices:
                    if not self.trigger_add:
                        image = add_pixel_pattern_backdoor(image, self.pattern_mask,self.location)
                    else:
                        # raise ValueError('not support additive trigger!')
                        if self.pixel_pattern is None:
                            self.pixel_pattern = ms.ops.full_like(image, 0)
                        image = image + self.pixel_pattern


        # bhi : image_list 2,50,50,3
        if self.party_num < 3:
            images = tuple(image for image in images_list)  # 3,3,50,50
        else:
            # images = ms.ops.stack(tuple(image for image in images_list), 0)  # 3,3,50,50
            images = tuple(image for image in images_list)  # 3,3,50,50
        # TODO not numpyarray
        # print("111",len(images),len(images[0]),len(images[0][0]),len(images[0][0][0]))
        return images, target, old_image


    def __len__(self) -> int:
        return len(self.data)


    def split_func(self, img):
        if self.party_num == 4:
            # split image into halves vertically for parties
            img_a, img_b1, img_b2, img_b3 = img[:, :16, :16], img[:, :16, 16:], img[:, 16:, :16], img[:, 16:, 16:]
            images_list = [img_a, img_b1, img_b2, img_b3]
        elif self.party_num == 8:
            img_a, img_b1, img_b2, img_b3, img_b4, img_b5, img_b6, img_b7 = img[:, :16, :8], img[:, :16, 8:16], \
                img[:, :16, 16:24], img[:, :16, 24:], img[:, 16:, :8], img[:, 16:, 8:16], img[:, 16:, 16:24], img[:,16:, 24:]
            images_list = [img_a, img_b1, img_b2, img_b3, img_b4, img_b5, img_b6, img_b7]
        elif self.party_num == 6:
            img_a, img_b1, img_b2, img_b3, img_b4, img_b5 = img[:, :16, :11], img[:, :16, 11:22], img[:, :16, 22:], \
                img[:, 16:, :11], img[:, 16:, 11:22], img[:, 16:, 22:]
            images_list = [img_a, img_b1, img_b2, img_b3, img_b4, img_b5]
        else:
            images_list = []
            length = 32 // self.party_num + 1
            for i in range(self.party_num):
                end = min((i + 1) * length, 32)
                images_list.append(img[:, :, i * length:end])
        return images_list
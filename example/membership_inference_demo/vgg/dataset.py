# Copyright 2020 Huawei Technologies Co., Ltd
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
dataset processing.
"""
import os
from mindspore.common import dtype as mstype
import mindspore.dataset as de
import mindspore.dataset.transforms.c_transforms as C
import mindspore.dataset.transforms.vision.c_transforms as vision


def vgg_create_dataset100(data_home, image_size, batch_size, rank_id=0, rank_size=1, repeat_num=1,
                          training=True, num_samples=None, shuffle=True):
    """Data operations."""
    de.config.set_seed(1)
    data_dir = os.path.join(data_home, "train")
    if not training:
        data_dir = os.path.join(data_home, "test")

    if num_samples is not None:
        data_set = de.Cifar100Dataset(data_dir, num_shards=rank_size, shard_id=rank_id,
                                      num_samples=num_samples, shuffle=shuffle)
    else:
        data_set = de.Cifar100Dataset(data_dir, num_shards=rank_size, shard_id=rank_id)

    input_columns = ["fine_label"]
    output_columns = ["label"]
    data_set = data_set.rename(input_columns=input_columns, output_columns=output_columns)
    data_set = data_set.project(["image", "label"])

    rescale = 1.0 / 255.0
    shift = 0.0

    # define map operations
    random_crop_op = vision.RandomCrop((32, 32), (4, 4, 4, 4))  # padding_mode default CONSTANT
    random_horizontal_op = vision.RandomHorizontalFlip()
    resize_op = vision.Resize(image_size)  # interpolation default BILINEAR
    rescale_op = vision.Rescale(rescale, shift)
    normalize_op = vision.Normalize((0.4465, 0.4822, 0.4914), (0.2010, 0.1994, 0.2023))
    changeswap_op = vision.HWC2CHW()
    type_cast_op = C.TypeCast(mstype.int32)

    c_trans = []
    if training:
        c_trans = [random_crop_op, random_horizontal_op]
    c_trans += [resize_op, rescale_op, normalize_op,
                changeswap_op]

    # apply map operations on images
    data_set = data_set.map(input_columns="label", operations=type_cast_op)
    data_set = data_set.map(input_columns="image", operations=c_trans)

    # apply repeat operations
    data_set = data_set.repeat(repeat_num)

    # apply shuffle operations
    # data_set = data_set.shuffle(buffer_size=1000)

    # apply batch operations
    data_set = data_set.batch(batch_size=batch_size, drop_remainder=True)

    return data_set

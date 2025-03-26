# Copyright 2021 Huawei Technologies Co., Ltd
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
"""data processing"""

import os
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as CV
import mindspore.dataset.transforms.c_transforms as C
from mindspore.dataset.vision import Inter
import mindspore.common.dtype as mstype


def generate_mnist_dataset(data_path, batch_size=32, repeat_size=1,
                           num_samples=None, num_parallel_workers=1, sparse=True):
    """
    create dataset for training or testing
    """
    # define dataset
    ds1 = ds.MnistDataset(data_path, num_samples=num_samples)

    # define operation parameters
    resize_height, resize_width = 32, 32
    rescale = 1.0 / 255.0
    shift = 0.0

    # define map operations
    resize_op = CV.Resize((resize_height, resize_width),
                          interpolation=Inter.LINEAR)
    rescale_op = CV.Rescale(rescale, shift)
    hwc2chw_op = CV.HWC2CHW()
    type_cast_op = C.TypeCast(mstype.int32)

    # apply map operations on images
    if not sparse:
        one_hot_enco = C.OneHot(10)
        ds1 = ds1.map(input_columns="label", operations=one_hot_enco,
                      num_parallel_workers=num_parallel_workers)
        type_cast_op = C.TypeCast(mstype.float32)
    ds1 = ds1.map(input_columns="label", operations=type_cast_op,
                  num_parallel_workers=num_parallel_workers)
    ds1 = ds1.map(input_columns="image", operations=resize_op,
                  num_parallel_workers=num_parallel_workers)
    ds1 = ds1.map(input_columns="image", operations=rescale_op,
                  num_parallel_workers=num_parallel_workers)
    ds1 = ds1.map(input_columns="image", operations=hwc2chw_op,
                  num_parallel_workers=num_parallel_workers)

    # apply DatasetOps
    buffer_size = 10000
    ds1 = ds1.shuffle(buffer_size=buffer_size)
    ds1 = ds1.batch(batch_size, drop_remainder=True)
    ds1 = ds1.repeat(repeat_size)

    return ds1


def vgg_create_dataset100(data_home, image_size, batch_size, rank_id=0, rank_size=1, repeat_num=1,
                          training=True, num_samples=None, shuffle=True):
    """Data operations."""
    ds.config.set_seed(1)
    data_dir = os.path.join(data_home, "train")
    if not training:
        data_dir = os.path.join(data_home, "test")

    if num_samples is not None:
        data_set = ds.Cifar100Dataset(data_dir, num_shards=rank_size, shard_id=rank_id,
                                      num_samples=num_samples, shuffle=shuffle)
    else:
        data_set = ds.Cifar100Dataset(data_dir, num_shards=rank_size, shard_id=rank_id)

    input_columns = ["fine_label"]
    output_columns = ["label"]
    data_set = data_set.rename(input_columns=input_columns, output_columns=output_columns)
    data_set = data_set.project(["image", "label"])

    rescale = 1.0 / 255.0
    shift = 0.0

    # define map operations
    random_crop_op = CV.RandomCrop((32, 32), (4, 4, 4, 4))  # padding_mode default CONSTANT
    random_horizontal_op = CV.RandomHorizontalFlip()
    resize_op = CV.Resize(image_size)  # interpolation default BILINEAR
    rescale_op = CV.Rescale(rescale, shift)
    normalize_op = CV.Normalize((0.4465, 0.4822, 0.4914), (0.2010, 0.1994, 0.2023))
    changeswap_op = CV.HWC2CHW()
    type_cast_op = C.TypeCast(mstype.int32)

    c_trans = []
    if training:
        c_trans = [random_crop_op, random_horizontal_op]
    c_trans += [resize_op, rescale_op, normalize_op,
                changeswap_op]

    # apply map operations on images
    data_set = data_set.map(input_columns="label", operations=type_cast_op)
    data_set = data_set.map(input_columns="image", operations=c_trans)

    # apply shuffle operations
    data_set = data_set.shuffle(buffer_size=1000)

    # apply batch operations
    data_set = data_set.batch(batch_size=batch_size, drop_remainder=True)

    # apply repeat operations
    data_set = data_set.repeat(repeat_num)
    return data_set


def create_dataset_imagenet(path, batch_size=32, repeat_size=20, status="train", target="GPU"):
    image_ds = ds.ImageFolderDataset(path, decode=True)
    rescale = 1.0 / 255.0
    shift = 0.0
    cfg = {'num_classes': 10,
           'learning_rate': 0.002,
           'momentum': 0.9,
           'epoch_size': 30,
           'batch_size': batch_size,
           'buffer_size': 1000,
           'image_height': 224,
           'image_width': 224,
           'save_checkpoint_steps': 1562,
           'keep_checkpoint_max': 10,
           'device': target,
           'status': status}
    resize_op = CV.Resize((cfg['image_height'], cfg['image_width']))
    rescale_op = CV.Rescale(rescale, shift)
    normalize_op = CV.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    random_crop_op = CV.RandomCrop([32, 32], [4, 4, 4, 4])
    random_horizontal_op = CV.RandomHorizontalFlip()
    channel_swap_op = CV.HWC2CHW()
    typecast_op = C.TypeCast(mstype.int32)
    image_ds = image_ds.map(input_columns="label", operations=typecast_op, num_parallel_workers=6)

    image_ds = image_ds.map(input_columns="image", operations=random_crop_op, num_parallel_workers=6)
    image_ds = image_ds.map(input_columns="image", operations=random_horizontal_op, num_parallel_workers=6)
    image_ds = image_ds.map(input_columns="image", operations=resize_op, num_parallel_workers=6)
    image_ds = image_ds.map(input_columns="image", operations=rescale_op, num_parallel_workers=6)
    image_ds = image_ds.map(input_columns="image", operations=normalize_op, num_parallel_workers=6)
    image_ds = image_ds.map(input_columns="image", operations=channel_swap_op, num_parallel_workers=6)

    image_ds = image_ds.shuffle(buffer_size=cfg['buffer_size'])
    image_ds = image_ds.repeat(repeat_size)
    return image_ds


def create_dataset_cifar(data_path, image_height, image_width, repeat_num=1, training=True):
    """
    create data for next use such as training or inferring
    """
    usage = "train" if training else "test"
    cifar_ds = ds.Cifar10Dataset(dataset_dir=data_path, usage=usage)
    resize_height = image_height  # 224
    resize_width = image_width  # 224
    rescale = 1.0 / 255.0
    shift = 0.0
    batch_size = 32
    # define map operations
    random_crop_op = CV.RandomCrop((32, 32), (4, 4, 4, 4))  # padding_mode default CONSTANT
    random_horizontal_op = CV.RandomHorizontalFlip()
    resize_op = CV.Resize((resize_height, resize_width))  # interpolation default BILINEAR
    rescale_op = CV.Rescale(rescale, shift)
    normalize_op = CV.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    changeswap_op = CV.HWC2CHW()
    type_cast_op = C.TypeCast(mstype.int32)
    c_trans = []
    if training:
        c_trans = [random_crop_op, random_horizontal_op]
    c_trans += [resize_op, rescale_op, normalize_op,
                changeswap_op]
    # apply map operations on images
    cifar_ds = cifar_ds.map(operations=type_cast_op, input_columns="label")
    cifar_ds = cifar_ds.map(operations=c_trans, input_columns="image")
    # apply shuffle operations
    cifar_ds = cifar_ds.shuffle(buffer_size=10)
    # apply batch operations
    cifar_ds = cifar_ds.batch(batch_size=batch_size, drop_remainder=True)
    # apply repeat operations
    cifar_ds = cifar_ds.repeat(repeat_num)
    return cifar_ds


def generate_dataset_cifar(data_path, batch_size, usage, repeat_num=1):
    """
    create data for next use such as training or inferring
    """
    cifar_ds = ds.Cifar10Dataset(data_path, usage=usage)
    resize_height = 32
    resize_width = 32
    rescale = 1.0 / 255.0
    shift = 0.0
    batch_size = batch_size
    # define map operations
    resize_op = CV.Resize((resize_height, resize_width))  # interpolation default BILINEAR
    rescale_op = CV.Rescale(rescale, shift)
    changeswap_op = CV.HWC2CHW()
    type_cast_op = C.TypeCast(mstype.int32)
    c_trans = []
    c_trans += [resize_op, rescale_op,
                changeswap_op]
    # apply map operations on images
    cifar_ds = cifar_ds.map(operations=type_cast_op, input_columns="label")
    cifar_ds = cifar_ds.map(operations=c_trans, input_columns="image")
    # apply shuffle operations
    cifar_ds = cifar_ds.shuffle(buffer_size=10)
    # apply batch operations
    cifar_ds = cifar_ds.batch(batch_size=batch_size, drop_remainder=True)
    # apply repeat operations
    cifar_ds = cifar_ds.repeat(repeat_num)
    return cifar_ds

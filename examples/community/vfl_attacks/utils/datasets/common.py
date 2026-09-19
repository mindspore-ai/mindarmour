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
Define data loaders for Vertical Federated Learning (VFL).

This module provides functions to create data loaders specifically designed for VFL scenarios.
"""
import mindspore as ms
from datasets.image_dataset import ImageDataset

__all__ = ["generate_dataloader"]

def generate_dataloader(data_list, batch_size, transform=None, shuffle=True, backdoor_indices=None,
                        trigger=None, trigger_add=None, source_indices=None, half=16):
    """
    Generate loader from dataset.

    Args:
        data_list (tuple): Contains X and Y.
        batch_size (int): Batch size of the loader.
        transform: Transform of the loader.
        shuffle (bool): Whether to shuffle the loader.
        backdoor_indices (ndarray): Indices of backdoor samples in normal dataset.
                          Adds trigger when loading data if index is in backdoor_indices.
        trigger (str): Controls whether to add a pixel trigger when loading the data loader.
        half (int): An integer specifying the size of a data subset
    Returns:
        DataLoader: The generated loader.
    """
    x, y = data_list

    ImageDatasetWithIndices = _image_dataset_with_indices(ImageDataset)
    # Split x into halves for parties when loading data, only support two parties.
    ds = ImageDatasetWithIndices(x, ms.tensor(y, ms.int32),
                                 transform=transform,
                                 backdoor_indices=backdoor_indices,
                                 half=half, trigger=trigger, trigger_add=trigger_add, source_indices=source_indices)
    dl = ms.dataset.GeneratorDataset(source=ds, shuffle=shuffle, column_names=['image','target','old_imgb', 'indice'])
    dl = dl.batch(batch_size, drop_remainder=False)
    return dl


def _image_dataset_with_indices(cls):
    """
    Build dataset class that can output x, y, and index when loading data based on cls, used for image dataset.

    Args:
        cls: The original dataset class.

    Returns:
        type: New dataset class.
    """
    def __getitem__(self, index):
        X_data, target, original = cls.__getitem__(self, index)
        return X_data, target, original, index

    type_of_cls =  type(cls.__name__, (cls,), {
        '__getitem__': __getitem__,
    })
    return type_of_cls

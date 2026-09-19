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
This module provides functions to select samples from a dataset.

Functions:
    _get_target_indices chooses the samples of the specified category.
    _get_random_indices randomly selects samples.
"""
import numpy as np

def get_target_indices(labels, target_label, size, backdoor_indices=None):
    """
    Get indices with a specified size of the target label.

    Args:
        labels (ndarray): Array of labels in the dataset.
        target_label (int): The target label to filter.
        size (int): The number of indices to return.

    Returns:
        ndarray: An array of indices with the specified size of the target label.
    """
    indices = np.where(labels == target_label)[0]
    indices = np.setdiff1d(indices, backdoor_indices)
    np.random.shuffle(indices)
    result = indices[:size]
    return result


def get_random_indices(target_length, all_length):
    """
    Generate random indices.

    Args:
        target_length (int): The length of the target indices to generate.
        all_length (int): The total length of all indices available.

    Returns:
        ndarray: An array of random indices.
    """
    all_indices = np.arange(all_length)
    indices = np.random.choice(all_indices, target_length, replace=False)
    return indices

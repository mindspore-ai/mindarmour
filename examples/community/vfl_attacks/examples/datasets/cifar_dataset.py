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
Load the dataset and construct the dataloader.

This module provides functions to create data loaders for the CIFAR-10 and CIFAR-100 datasets,
including train data loader, test data loader, and backdoor test data loader.
"""
import os
import pickle
import numpy as np
import mindspore as ms
from mindspore.dataset import vision
from utils.datasets.common import generate_dataloader
from examples.datasets.functions import get_random_indices, get_target_indices
from examples.common.constants import data_path

# Transform for CIFAR train dataset.
train_transform = ms.dataset.transforms.Compose([
    vision.ToTensor()
])

# Transform for CIFAR test dataset.
test_transform = ms.dataset.transforms.Compose([
    vision.ToTensor()
])

def _get_labeled_data_with_2_party(data_dir, dataset, dtype="train", num_samples=None):
    """
    Read data from a local file.

    Args:
        data_dir (str): Directory path of the local file.
        dataset (str): Dataset name, supported values are 'cifar10' and 'cifar100'.
        dtype (str): Type of data to read, either "Train" or "Test".

    Returns:
        tuple: A tuple containing the data X and the labels Y.
    """
    if dataset == 'cifar10':
        data_dir = data_dir + 'cifar-10-batches-py/'
        train_list = [
            'data_batch_1',
            'data_batch_2',
            'data_batch_3',
            'data_batch_4',
            'data_batch_5']
        test_list = ['test_batch']
        all_data = []
        targets = []
        downloaded_list = train_list if dtype == 'train' else test_list
        for file_name in downloaded_list:
            file_path = os.path.join(data_dir, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                all_data.append(entry['data'])
                if 'labels' in entry:
                    targets.extend(entry['labels'])
                else:
                    targets.extend(entry['fine_labels'])
        all_data = np.vstack(all_data).reshape(-1, 3, 32, 32)
        targets = np.array(targets)
        if num_samples is not None:
            indices = get_random_indices(num_samples, len(all_data))
            datas, labels = all_data[indices], targets[indices]
        else:
            datas, labels = all_data, targets
    else:
        filename = data_dir + 'cifar-100-python/' + dtype
        with open(filename, 'rb') as f:
            datadict = pickle.load(f, encoding='latin1')
            x = datadict['data']
            all_data = x.reshape(-1, 3, 32, 32)
            targets = datadict['fine_labels']
            targets = np.array(targets)
        if num_samples is not None:
            indices = get_random_indices(num_samples, len(all_data))
            datas, labels = all_data[indices], targets[indices]
        else:
            datas, labels = all_data, targets

    return datas, labels


def _load_two_party_data(data_dir, args):
    """
    Get data from a local dataset, supporting only two parties.

    Args:
        data_dir (str): Path of the local dataset.
        args (dict): Configuration.

    Returns:
        tuple: A tuple containing the following data:
            X_train: Normal train features.
            y_train: Normal train labels.
            X_test: Normal test features.
            y_test: Normal test labels.
            backdoor_X_test: Backdoor test features.
            backdoor_y_test: Backdoor test labels.
            backdoor_indices_train: Indices of backdoor samples in the normal train dataset.
            backdoor_target_indices: Indices of backdoor labels in the normal train dataset.
    """
    print("# load_two_party_data")
    n_train = args['target_train_size']
    n_test = args['target_test_size']
    if n_train == -1:
        n_train = None
    if n_test == -1:
        n_test = None

    x_train, y_train = _get_labeled_data_with_2_party(data_dir=data_dir,
                                         dataset=args['dataset'],
                                         dtype='train',
                                         num_samples=n_train)

    x_test, y_test = _get_labeled_data_with_2_party(data_dir=data_dir,
                                                   dataset=args['dataset'],
                                                   dtype='test',
                                                   num_samples=n_test)

    # Randomly select samples of other classes from normal train dataset as backdoor samples.
    train_indices = np.where(y_train != args['backdoor_label'])[0]
    backdoor_indices_train = np.random.choice(train_indices, args['backdoor_train_size'], replace=False)

    # Randomly select samples of other classes from normal test dataset to generate backdoor test dataset.
    test_indices = np.where(y_test != args['backdoor_label'])[0]
    backdoor_indices_test = np.random.choice(test_indices, args['backdoor_test_size'], replace=False)
    backdoor_x_test, backdoor_y_test = x_test[backdoor_indices_test], \
                                       y_test[backdoor_indices_test]
    backdoor_y_test = np.full_like(backdoor_y_test, args['backdoor_label'])

    # Randomly select samples of backdoor label in normal train dataset, for gradient-replacement.
    backdoor_target_indices = get_target_indices(y_train, args['backdoor_label'], args['backdoor_train_size'])

    print(f"y_train.shape: {y_train.shape}")
    print(f"y_test.shape: {y_test.shape}")
    print(f"backdoor_y_test.shape: {backdoor_y_test.shape}")

    return x_train, y_train, x_test, y_test, backdoor_x_test, backdoor_y_test, \
           backdoor_indices_train, backdoor_target_indices


def get_cifar_dataloader(args):
    """
    Generate loaders for the CIFAR dataset, supporting CIFAR-10 and CIFAR-100.

    Args:
        args (dict): Configuration.

    Returns:
        tuple: A tuple containing the following data loaders:
            train_dl: Loader for the normal train dataset.
            test_dl: Loader for the normal test dataset.
            backdoor_test_dl: Loader for the backdoor test dataset, containing only backdoor samples,
                            used for ASR evaluation.
            backdoor_indices: Indices of backdoor samples in the normal train dataset.
            backdoor_target_indices: Indices of backdoor labels in the normal train dataset,
                            used by Gradient-Replacement.
    """
    result = _load_two_party_data(data_path, args)
    x_train, y_train, x_test, y_test, backdoor_x_test, backdoor_y_test, \
    backdoor_indices, backdoor_target_indices = result

    batch_size = args['target_batch_size']
    # Get loader of normal train dataset, used by normal training.
    train_dl = generate_dataloader((x_train, y_train), batch_size, train_transform, shuffle=True)
    # GFet loader of normal test dataset, used to evaluate main task accuracy.
    test_dl = generate_dataloader((x_test, y_test), batch_size, test_transform, shuffle=False)

    backdoor_test_dl = None
    if args['backdoor'] != 'no':
        # Get loader of backdoor test dataset, used to evaluate backdoor task accuracy.
        backdoor_test_dl = generate_dataloader((backdoor_x_test, backdoor_y_test), batch_size, test_transform,
                                               shuffle=False,
                                               backdoor_indices=np.arange(args['backdoor_test_size']),
                                               trigger=args['trigger'], trigger_add=args['trigger_add'])

    if args['backdoor'] == 'g_r':
        # Get loader of train dataset used by Gradient-Replacement, containing backdoor features and normal labels.
        train_dl = generate_dataloader((x_train, y_train), batch_size, train_transform,
                                           shuffle=True,
                                           backdoor_indices=backdoor_indices)

    return train_dl, test_dl, backdoor_test_dl, backdoor_indices, backdoor_target_indices
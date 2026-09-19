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
import logging

from MindsporeCode.datasets.cifar_dataset import get_cifar_dataloader
from datasets.cinic_dataset import get_cinic_dataloader
from MindsporeCode.datasets.bhi_dataset import get_bhi_dataloader
from MindsporeCode.datasets.criteo_dataset import get_criteo_dataloader

def get_dataloader(args):
    """
    generate data loader according to dataset name

    :param args: configuration
    :return: data loader
    """
    if 'cifar' in args['dataset']:
        return get_cifar_dataloader(args)
    elif args['dataset'] == 'cinic':
        return get_cinic_dataloader(args)
    elif args['dataset'] == 'bhi':
        return get_bhi_dataloader(args)
    elif args['dataset'] == 'criteo':
        return get_criteo_dataloader(args)


def get_backdoor_target_index(train_loader, backdoor_indices, args):
    """
    get index of a normal input labeled backdoor class in training dataset, used for gradient-replacement backdoor

    :param train_loader: loader of training dataset
    :param backdoor_indices: indices of backdoor samples
    :param args: configuration
    :return: index of a normal input labeled backdoor class
    """
    for (_, _), labels, indices in train_loader:
        for label, index in zip(labels, indices):
            if label == args['backdoor_label'] and index not in backdoor_indices:
                logging.info('backdoor target index: {}'.format(index))
                return index.item()
    return None


def get_num_classes(dataset):
    """
    get classes number of the target dataset

    :param str dataset: target dataset name
    :return: classes number of the target dataset
    """
    data_dict = {
        'cifar10': 10,
        'cifar100': 100,
        'cinic': 10,
        'bhi':2,
        'criteo': 2
    }
    return data_dict[dataset]




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
import copy
import logging

import numpy as np
import mindspore as ms
from sklearn.model_selection import train_test_split
import pandas as pd

from MindsporeCode.datasets.common import train_label_split, get_labeled_loader, get_target_indices, image_dataset_with_indices, get_random_indices
from MindsporeCode.datasets.multi_tabular_dataset import MultiTabularDataset
from MindsporeCode.datasets.tabular_dataset import TabularDataset


def get_labeled_data(data_path):
    """
    read data from local file, including training and testing

    :param str data_dir: dir path of the csv file
    :return: tuple containing X and Y
    """
    total_samples_num = 1e5
    df_labels = pd.read_csv(data_path, nrows=total_samples_num, usecols=['label'])
    labels = df_labels.astype('long').values.reshape(-1)
    df_features = pd.read_csv(data_path, nrows=total_samples_num, usecols=lambda x: 'feat' in x)
    features = df_features.astype('long').values
    return features, labels


def load_parties_data(data_dir, args):
    """
    get data from local dataset

    :param data_dir: path of local dataset
    :param args: configuration
    :return: tuple contains:
        (1) X_train: normal train features;
        (2) y_train: normal train labels;
        (3) X_test: normal test features;
        (4) y_test: normal test labels;
        (5) backdoor_y_train: backdoor train labels;
        (6) backdoor_X_test: backdoor test features;
        (7) backdoor_y_test: backdoor test labels;
        (8) backdoor_indices_train: indices of backdoor samples in normal train dataset;
        (9) backdoor_target_indices: indices of backdoor label in normal train dataset;
        (10) train_labeled_indices: indices of labeled samples in normal train dataset;
        (11) train_unlabeled_indices: indices of unlabeled samples in normal train dataset
    """
    logging.info("# load_parties_data")
    # read data from local file
    X, y = get_labeled_data(data_path=data_dir)

    # split normal dataset for train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
                                                        test_size=0.2,
                                                        random_state=1)

    n_train = args['target_train_size']
    n_test = args['target_test_size']
    if n_train != -1:
        indices = get_random_indices(n_train, len(X))
        X_train, y_train = X[indices], y[indices]
    else:
        X_train, y_train = X, y
    if n_test != -1:
        indices = get_random_indices(n_test, len(X_test))
        X_test, y_test = X_test[indices], y_test[indices]

    # randomly select samples of other classes from normal train dataset as backdoor samples to generate backdoor train dataset
    train_indices = np.where(y_train != args['backdoor_label'])[0]
    num_train_choices = min(len(train_indices), args['backdoor_train_size'])
    if num_train_choices < args['backdoor_train_size']:
        logging.warning(
            f"[Warning] backdoor_train_size ({args['backdoor_train_size']}) is larger than available train samples ({len(train_indices)}). Using {num_train_choices} instead.")
    backdoor_indices_train = np.random.choice(train_indices, num_train_choices, replace=False)
    backdoor_y_train = copy.deepcopy(y_train)
    backdoor_y_train[backdoor_indices_train] = args['backdoor_label']

    # randomly select samples of other classes from normal test dataset to generate backdoor test dataset
    test_indices = np.where(y_test != args['backdoor_label'])[0]
    num_test_choices = min(len(test_indices), args['backdoor_test_size'])
    if num_test_choices < args['backdoor_test_size']:
        logging.warning(
            f"[Warning] backdoor_test_size ({args['backdoor_test_size']}) is larger than available test samples ({len(test_indices)}). Using {num_test_choices} instead.")
    backdoor_indices_test = np.random.choice(test_indices, num_test_choices, replace=False)
    backdoor_X_test, backdoor_y_test = X_test[backdoor_indices_test], \
        y_test[backdoor_indices_test]
    backdoor_y_test = np.full_like(backdoor_y_test, args['backdoor_label'])

    # split labeled and unlabeled samples in normal train dataset, for LR-BA
    train_labeled_indices, train_unlabeled_indices = \
        train_label_split(y_train, args['train_label_size'], args['num_classes'],
                          args['train_label_non_iid'], args['backdoor_label'], args['train_label_fix_backdoor'])

    # randomly select samples of backdoor label in normal train dataset, for gradient-replacement
    backdoor_target_indices = get_target_indices(y_train, args['backdoor_label'], args['train_label_size'])

    logging.info("y_train.shape: {}".format(y_train.shape))
    logging.info("y_test.shape: {}".format(y_test.shape))
    logging.info("backdoor_y_test.shape: {}".format(backdoor_y_test.shape))
    return X_train, y_train, X_test, y_test, backdoor_y_train, backdoor_X_test, backdoor_y_test, \
        backdoor_indices_train, backdoor_target_indices, train_labeled_indices, train_unlabeled_indices


def generate_dataloader(args, data_list, batch_size, transform=None, shuffle=True, backdoor_indices=None, trigger=None,
                        trigger_add=None, source_indices=None):
    """
    generate loader from dataset

    :param tuple data_list: contains X and Y
    :param int batch_size: batch of loader
    :param transform: transform of loader
    :param bool shuffle: whether to shuffle loader
    :param backdoor_indices: indices of backdoor samples in normal dataset, add trigger when loading data if index is in backdoor_indices
    :param trigger: control whether add pixel trigger when load dataloader
    :return: loader
    """
    X, y = data_list

    if args['n_passive_party'] > 1:
        MultiImageDatasetWithIndices = image_dataset_with_indices(MultiTabularDataset)
        party_num = args['n_passive_party'] + 1

        ds = MultiImageDatasetWithIndices(X, ms.Tensor(y, dtype=ms.int32),
                                          transform=transform,
                                          backdoor_indices=backdoor_indices,
                                          party_num=party_num,
                                          trigger=trigger, trigger_add=trigger_add, source_indices=source_indices,
                                          adversary=args['adversary'])
    else:
        ImageDatasetWithIndices = image_dataset_with_indices(TabularDataset)
        # split x into halves for parties when loading data, only support two parties
        ds = ImageDatasetWithIndices(X, ms.Tensor(y, dtype=ms.int32),
                                     backdoor_indices=backdoor_indices,
                                     half=2 ** 12, trigger=trigger, trigger_add=trigger_add,
                                     source_indices=source_indices)

    dl = ms.dataset.GeneratorDataset(source=ds, column_names=['image','target','old_imgb', 'indice'], shuffle=shuffle)

    dl = dl.batch(batch_size, drop_remainder=False)
    return dl


def get_criteo_dataloader(args):
    """
    generate loader of criteo dataset

    :param args: configuration
    :return: tuple contains:
        (1) train_dl: loader of normal train dataset;
        (2) test_dl: loader of normal test dataset;
        (3) backdoor_train_dl: loader of backdoor train dataset, including normal and backdoor samples, used by data poisoning
        (4) backdoor_test_dl: loader of backdoor test dataset, only including backdoor samples, used to evaluate ASR
        (5) backdoor_indices: indices of backdoor samples in normal train dataset;
        (6) backdoor_target_indices: indices of backdoor label in normal train dataset, used by Gradient-Replacement
        (7) labeled_dl: loader of labeled samples in normal train dataset, used by LR-BA;
        (8) unlabeled_dl: loader of unlabeled samples in normal train dataset, used by LR-BA
    """
    party_num = args['n_passive_party'] + 1
    result = load_parties_data(data_dir="../../data/Criteo/criteo.csv", args=args)
    X_train, y_train, X_test, y_test, backdoor_y_train, backdoor_X_test, backdoor_y_test, \
        backdoor_indices, backdoor_target_indices, train_labeled_indices, train_unlabeled_indices = result

    batch_size = args['target_batch_size']
    # get loader of normal train dataset, used by normal training and LR-BA
    train_dl = generate_dataloader(args, (X_train, y_train), batch_size, None, shuffle=True)
    # get loader of normal test dataset, used to evaluate main task accuracy
    test_dl = generate_dataloader(args, (X_test, y_test), batch_size, None, shuffle=False)

    # get loader of backdoor train dataset, used by data poisoning attack
    backdoor_train_dl = generate_dataloader(args, (X_train, backdoor_y_train), batch_size, None,
                                            shuffle=True,
                                            backdoor_indices=backdoor_indices)
    # get loader of backdoor test dataset, used to evaluate backdoor task accuracy
    backdoor_test_dl = generate_dataloader(args, (backdoor_X_test, backdoor_y_test), batch_size, None,
                                           shuffle=False,
                                           backdoor_indices=np.arange(args['backdoor_test_size']),
                                           trigger=args['trigger'], trigger_add=args['trigger_add'])

    labeled_data, unlabeled_data = get_labeled_loader(train_dataset=(X_train, y_train),
                                                      labeled_indices=train_labeled_indices,
                                                      unlabeled_indices=train_unlabeled_indices,
                                                      args=args)
    labeled_dl = generate_dataloader(args, labeled_data,
                                     batch_size=min(len(train_labeled_indices), args['lr_ba_top_batch_size']),
                                     shuffle=True)
    unlabeled_dl = generate_dataloader(args, unlabeled_data,
                                       batch_size=min(len(train_labeled_indices), args['lr_ba_top_batch_size']),
                                       shuffle=True)

    # get loader of train dataset used by Gradient-Replacement, containing backdoor features and normal labels
    g_r_train_dl = generate_dataloader(args, (X_train, y_train), batch_size, None,
                                       shuffle=True,
                                       backdoor_indices=backdoor_indices)

    if args['backdoor'] == 'villain':
        villain_train_dl = generate_dataloader(args, (X_train, y_train), batch_size, None,
                                               shuffle=True,
                                               backdoor_indices=backdoor_target_indices, trigger=args['trigger'],
                                               trigger_add=args['trigger_add'])
    else:
        villain_train_dl = None

    return train_dl, test_dl, backdoor_train_dl, backdoor_test_dl, g_r_train_dl, \
        backdoor_indices, backdoor_target_indices, labeled_dl, unlabeled_dl, villain_train_dl

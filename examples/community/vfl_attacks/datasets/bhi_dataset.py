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
import os
import pickle

import numpy as np
import mindspore as ms
from mindspore.dataset import vision

from sklearn.model_selection import train_test_split
from MindsporeCode.datasets.common import train_label_split, get_labeled_loader, get_target_indices, image_dataset_with_indices, get_random_indices
from MindsporeCode.datasets.multi_image_dataset import MultiImageDataset
from MindsporeCode.common.constants import data_path
# transform for BHI images
normalize = vision.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5],is_hwc = True)
transform = ms.dataset.transforms.Compose([
    normalize
])

def get_labeled_data(data_dir, party_num):
    """
    read data from local file, including training and testing

    :param str data_dir: dir path of local file
    :param int party_num: parties number
    :return: tuple containing X and Y
    """
    file_path = os.path.join(data_dir, 'data.pkl')
    with open(file_path, 'rb') as f:
        entry = pickle.load(f, encoding='latin1')
        img_data = entry['data']  # [345910,50,50,3]
        targets = entry['labels']
    targets = np.array(targets)
    # split data by labels 0 and 1
    zero_indices = np.where(targets == 0)[0]  # 209202
    one_indices = np.where(targets == 1)[0]  # 136708

    # divide data evenly into parties
    groups_num_zero = int(len(zero_indices) / party_num)
    groups_num_one = int(len(one_indices) / party_num)
    path_groups_zero = []
    path_groups_one = []
    for group_zore_id in range(groups_num_zero):
        path_groups_zero.append(
            img_data[zero_indices[group_zore_id * party_num: group_zore_id * party_num + party_num]])
    for group_one_id in range(groups_num_one):
        path_groups_one.append(img_data[one_indices[group_one_id * party_num: group_one_id * party_num + party_num]])
    path_groups_zero.extend(path_groups_one)
    path_groups = path_groups_zero

    # generate labels of parties data
    labels_group = [0] * groups_num_zero
    labels_group.extend([1] * groups_num_one)

    return np.array(path_groups), np.array(labels_group)

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
    party_num = args['n_passive_party']+1
    logging.info("# load_parties_data")
    # read data from local file
    X, y = get_labeled_data(data_dir=data_dir, party_num=party_num)

    # split normal dataset for train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
                                                        test_size=0.2,
                                                        random_state=1)

    n_train = args['target_train_size']
    n_test = args['target_test_size']

    if n_train != -1:
        indices = get_random_indices(n_train, len(X_train))
        X_train, y_train = X_train[indices], y_train[indices]
    if n_test != -1:
        indices = get_random_indices(n_test, len(X_test))
        X_test, y_test = X_test[indices], y_test[indices]

    # randomly select samples of other classes from normal train dataset as backdoor samples to generate backdoor train dataset
    train_indices = np.where(y_train != args['backdoor_label'])[0]
    backdoor_indices_train = np.random.choice(train_indices, args['backdoor_train_size'], replace=False)
    backdoor_y_train = copy.deepcopy(y_train)
    backdoor_y_train[backdoor_indices_train] = args['backdoor_label']

    # randomly select samples of other classes from normal test dataset to generate backdoor test dataset
    test_indices = np.where(y_test != args['backdoor_label'])[0]
    backdoor_indices_test = np.random.choice(test_indices, args['backdoor_test_size'], replace=False)
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

def generate_dataloader(args, data_list, batch_size, transform=None, shuffle=True, backdoor_indices=None, trigger=None, trigger_add=None, source_indices=None):
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
    MultiImageDatasetWithIndices = image_dataset_with_indices(MultiImageDataset)
    party_num = args['n_passive_party'] + 1
    ds = MultiImageDatasetWithIndices(X, ms.tensor(y,ms.int32),
                                      transform=transform,
                                      backdoor_indices=backdoor_indices,
                                      party_num=party_num,
                                      trigger=trigger, trigger_add=trigger_add, source_indices=source_indices, adversary=args['adversary'])

    dl = ms.dataset.GeneratorDataset(source=ds, shuffle=shuffle, column_names=['image','target','old_imgb', 'indice'])
    dl = dl.batch(batch_size, drop_remainder=False)
    return dl

def get_bhi_dataloader(args):
    """
    generate loader of BHI dataset

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
    # get dataset
    result = load_parties_data(data_dir=data_path+"BHI/", args=args)
    X_train, y_train, X_test, y_test, backdoor_y_train, backdoor_X_test, backdoor_y_test, \
    backdoor_indices, backdoor_target_indices, train_labeled_indices, train_unlabeled_indices = result

    batch_size = args['target_batch_size']
    # get loader of normal train dataset, used by normal training and LR-BA
    train_dl = generate_dataloader(args, (X_train, y_train), batch_size, transform, shuffle=True)
    # get loader of normal test dataset, used to evaluate main task accuracy
    test_dl = generate_dataloader(args, (X_test, y_test), batch_size, transform, shuffle=False)

    # get loader of backdoor train dataset, used by data poisoning attack
    backdoor_train_dl = generate_dataloader(args, (X_train, backdoor_y_train), batch_size, transform,
                                            shuffle=True,
                                            backdoor_indices=backdoor_indices)
    # get loader of backdoor test dataset, used to evaluate backdoor task accuracy
    backdoor_test_dl = generate_dataloader(args, (backdoor_X_test, backdoor_y_test), batch_size, transform,
                                           shuffle=False,
                                           backdoor_indices=np.arange(args['backdoor_test_size']),
                                           trigger=args['trigger'], trigger_add=args['trigger_add'])

    # get loader of labeled and unlabeled normal train dataset, used by LR-BA
    labeled_data, unlabeled_data = get_labeled_loader(train_dataset=(X_train, y_train),         ##########
                                                  labeled_indices=train_labeled_indices,
                                                  unlabeled_indices=train_unlabeled_indices,
                                                  args=args)
    labeled_dl = generate_dataloader(args, labeled_data, batch_size=min(len(train_labeled_indices), args['lr_ba_top_batch_size']), shuffle=True)
    unlabeled_dl = generate_dataloader(args, unlabeled_data, batch_size=min(len(train_labeled_indices), args['lr_ba_top_batch_size']), shuffle=True)
    
    # get loader of train dataset used by Gradient-Replacement, containing backdoor features and normal labels
    g_r_train_dl = generate_dataloader(args, (X_train, y_train), batch_size, transform,
                                       shuffle=True,
                                       backdoor_indices=backdoor_indices)

    if args['backdoor'] == 'villain':
        villain_train_dl = generate_dataloader(args, (X_train, y_train), batch_size, transform,
                                           shuffle=True,
                                           backdoor_indices=backdoor_target_indices, trigger=args['trigger'],
                                           trigger_add=args['trigger_add'])
        # villain_train_dl = villain_train_dl.create_tuple_iterator()
        
    else:
        villain_train_dl = None

    return train_dl, test_dl, backdoor_train_dl, backdoor_test_dl, g_r_train_dl, \
        backdoor_indices, backdoor_target_indices, labeled_dl, unlabeled_dl, villain_train_dl
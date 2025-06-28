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
import random

import numpy as np
import mindspore as ms
from mindspore.dataset import vision
import mindspore.dataset as ds

from MindsporeCode.datasets.common import train_label_split, get_random_indices, \
    get_labeled_loader, get_target_indices, image_dataset_with_indices, poison_image_dataset
from MindsporeCode.datasets.image_dataset import ImageDataset
from MindsporeCode.datasets.multi_image_dataset import MultiImageDataset
from PIL import Image
import os
import pickle
from MindsporeCode.common.constants import data_path

# transform for CIFAR train dataset
train_transform = ms.dataset.transforms.Compose([
    vision.ToTensor(),

])

# transform for CIFAR test dataset
test_transform = ms.dataset.transforms.Compose([
    vision.ToTensor(),

])

def get_labeled_data_with_2_party(data_dir, dataset, dtype="train", num_samples=None):
    """
    read data from local file

    :param data_dir: dir path of local file
    :param str dataset: dataset name, support cifar10 and cifar100
    :param str dtype: read "Train" or "Test" data
    :return: tuple containing X and Y
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
        # now load the picked numpy arrays
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
            X = datadict['data']
            all_data = X.reshape(-1, 3, 32, 32)
            targets = datadict['fine_labels']
            targets = np.array(targets)
        if num_samples is not None:
            indices = get_random_indices(num_samples, len(all_data))
            datas, labels = all_data[indices], targets[indices]
        else:
            datas, labels = all_data, targets

    return datas, labels


def load_two_party_data(data_dir, args):
    """
    get data from local dataset, only support two parties

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
        (12) backdoor_X_train: train dataset that backdoor_target_indices sample had been replaced with backdoor_indices sample in adv
    """
    logging.info("# load_two_party_data")
    n_train = args['target_train_size']
    n_test = args['target_test_size']
    if n_train == -1:
        n_train = None
    if n_test == -1:
        n_test = None
    # read train data from local file
    X_train, y_train = get_labeled_data_with_2_party(data_dir=data_dir,
                                         dataset=args['dataset'],
                                         dtype='train',
                                         num_samples=n_train)

    # read test data from local file
    X_test, y_test = get_labeled_data_with_2_party(data_dir=data_dir,
                                                   dataset=args['dataset'],
                                                   dtype='test',
                                                   num_samples=n_test)

    # randomly select samples of other classes from normal train dataset as backdoor samples to generate backdoor train dataset
    train_indices = np.where(y_train != args['backdoor_label'])[0]
    num_train_choice = min(len(train_indices), args['backdoor_train_size'])
    if num_train_choice < args['backdoor_train_size']:
        logging.warning("backdoor train size is larger than normal train size, use normal train size")
    backdoor_indices_train = np.random.choice(train_indices, num_train_choice, replace=False)
    backdoor_y_train = copy.deepcopy(y_train)
    backdoor_y_train[backdoor_indices_train] = args['backdoor_label']

    # randomly select samples of other classes from normal test dataset to generate backdoor test dataset
    test_indices = np.where(y_test != args['backdoor_label'])[0]
    num_test_choice = min(len(test_indices), args['backdoor_test_size'])
    if num_test_choice < args['backdoor_test_size']:
        logging.warning("backdoor test size is larger than normal test size, use normal test size")
    backdoor_indices_test = np.random.choice(test_indices, num_test_choice, replace=False)
    backdoor_X_test, backdoor_y_test = X_test[backdoor_indices_test], \
        y_test[backdoor_indices_test]
    backdoor_y_test = np.full_like(backdoor_y_test, args['backdoor_label'])

    # split labeled and unlabeled samples in normal train dataset, for LR-BA
    train_labeled_indices, train_unlabeled_indices = \
        train_label_split(y_train, args['train_label_size'], args['num_classes'],
                          args['train_label_non_iid'], args['backdoor_label'], args['train_label_fix_backdoor'])

    # randomly select samples of backdoor label in normal train dataset, for gradient-replacement
    backdoor_target_indices = get_target_indices(y_train, args['backdoor_label'], args['backdoor_train_size'])

    logging.info("y_train.shape: {}".format(y_train.shape))
    logging.info("y_test.shape: {}".format(y_test.shape))
    logging.info("backdoor_y_test.shape: {}".format(backdoor_y_test.shape))

    labeled_y_train = y_train[train_labeled_indices]
    temp = []
    for i in range(args['num_classes']):
        indices = np.where(labeled_y_train == i)[0]
        temp.append(len(indices))
    # logging.info('labeled labels sum: {}, all: {}'.format(np.sum(temp), temp))
    logging.info('labeled labels sum: {}'.format(np.sum(temp)))

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
    if args['n_passive_party'] > 1:
        MultiImageDatasetWithIndices = image_dataset_with_indices(MultiImageDataset)
        party_num = args['n_passive_party'] + 1
        ds = MultiImageDatasetWithIndices(X, ms.tensor(y, ms.int32),
                                          transform=transform,
                                          backdoor_indices=backdoor_indices,
                                          party_num=party_num,
                                          trigger=trigger, trigger_add=trigger_add, source_indices=source_indices, adversary=args['adversary'])
    else:
        ImageDatasetWithIndices = image_dataset_with_indices(ImageDataset)
        # split x into halves for parties when loading data, only support two parties
        ds = ImageDatasetWithIndices(X, ms.tensor(y, ms.int32),
                                     transform=transform,
                                     backdoor_indices=backdoor_indices,
                                     half=16, trigger=trigger, trigger_add=trigger_add, source_indices=source_indices)
    dl = ms.dataset.GeneratorDataset(source=ds, shuffle=shuffle, column_names=['image','target','old_imgb', 'indice'])
    dl = dl.batch(batch_size, drop_remainder=False)
    return dl


def get_cifar_dataloader(args):
    """
    generate loader of CIFAR dataset, support cifar10 and cifar100

    :param args: configuration
    :return: tuple contains:
        (1) train_dl: loader of normal train dataset;
        (2) test_dl: loader of normal test dataset;
        (3) backdoor_train_dl: loader of backdoor train dataset, including normal and backdoor samples, used by data poisoning
        (4) backdoor_test_dl: loader of backdoor test dataset, only including backdoor samples, used to evaluate ASR
        (5) g_r_train_dl: loader of train dataset used by Gradient-Replacement, containing backdoor features and normal labels
        (6) backdoor_indices: indices of backdoor samples in normal train dataset;
        (7) backdoor_target_indices: indices of backdoor label in normal train dataset, used by Gradient-Replacement
        (8) labeled_dl: loader of labeled samples in normal train dataset, used by LR-BA;
        (9) unlabeled_dl: loader of unlabeled samples in normal train dataset, used by LR-BA
    """
    # get dataset
    result = load_two_party_data(data_path, args)
    X_train, y_train, X_test, y_test, backdoor_y_train, backdoor_X_test, backdoor_y_test, \
    backdoor_indices, backdoor_target_indices, train_labeled_indices, train_unlabeled_indices = result

    batch_size = args['target_batch_size']
    # get loader of normal train dataset, used by normal training and LR-BA
    train_dl = generate_dataloader(args, (X_train, y_train), batch_size, train_transform, shuffle=True)
    # get loader of normal test dataset, used to evaluate main task accuracy
    test_dl = generate_dataloader(args, (X_test, y_test), batch_size, test_transform, shuffle=False)

    backdoor_train_dl = generate_dataloader(args, (X_train, backdoor_y_train), batch_size, train_transform,
                                            shuffle=True,
                                            backdoor_indices=backdoor_indices)
    # get loader of backdoor test dataset, used to evaluate backdoor task accuracy
    backdoor_test_dl = generate_dataloader(args, (backdoor_X_test, backdoor_y_test), batch_size, test_transform,
                                           shuffle=False,
                                           backdoor_indices=np.arange(args['backdoor_test_size']), trigger=args['trigger'], trigger_add=args['trigger_add'])

    # get loader of labeled and unlabeled normal train dataset, used by LR-BA
    labeled_data, unlabeled_data = get_labeled_loader(train_dataset=(X_train, y_train),
                                                  labeled_indices=train_labeled_indices,
                                                  unlabeled_indices=train_unlabeled_indices,
                                                  args=args)
    labeled_dl = generate_dataloader(args, labeled_data, batch_size=min(len(train_labeled_indices), args['lr_ba_top_batch_size']), shuffle=True)
    unlabeled_dl = generate_dataloader(args, unlabeled_data, batch_size=min(len(train_labeled_indices), args['lr_ba_top_batch_size']), shuffle=True)


    # get loader of train dataset used by Gradient-Replacement, containing backdoor features and normal labels
    g_r_train_dl = generate_dataloader(args, (X_train, y_train), batch_size, train_transform,
                                       shuffle=True,
                                       backdoor_indices=backdoor_indices)

    if args['backdoor'] == 'villain':
        villain_train_dl = generate_dataloader(args, (X_train, y_train), batch_size, train_transform,
                                           shuffle=True,
                                           backdoor_indices=backdoor_target_indices, trigger=args['trigger'],
                                           trigger_add=args['trigger_add'])
        # villain_train_dl = villain_train_dl.create_tuple_iterator()

    else:
        villain_train_dl = None


    return train_dl, test_dl, backdoor_test_dl, backdoor_train_dl, g_r_train_dl, \
           backdoor_indices, backdoor_target_indices, labeled_dl, unlabeled_dl, \
            villain_train_dl

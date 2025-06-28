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
    # vision.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616], is_hwc=False)  # CIFAR10
    # transforms.Normalize((0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761))  #CIFAR100
])

# transform for CIFAR test dataset
test_transform = ms.dataset.transforms.Compose([
    vision.ToTensor(),
    # vision.Normalize(mean=[0.4940, 0.4850, 0.4504], std=[0.2467, 0.2429, 0.2616], is_hwc=False)  # CIFAR10
    # transforms.Normalize((0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761))  #CIFAR100
])

def get_labeled_data_with_2_party(data_dir, dataset, dtype="train", num_samples=None):
    """
    read data from local file

    :param data_dir: dir path of local file
    :param str dataset: dataset name, support cifar10 and cifar100
    :param str dtype: read "Train" or "Test" data
    :return: tuple containing X and Y
    """
    # data_dir = data_dir + 'cifar-10-batches-bin/'
    # transform = train_transform if dtype == 'train' else test_transform
    # if dataset == 'cifar10':
    #     dataset = ds.Cifar10Dataset(dataset_dir=data_dir,usage=dtype, num_samples=num_samples)
    #     # dataset = dataset.map(transform)
    #     dataset = dataset.map(operations=[(lambda x: x.transpose(2, 0, 1).unsqueeze(0).asnumpy())])
    #
    # elif dataset == 'cifar100':
    #     dataset = ds.Cifar100Dataset(dataset_dir=data_dir,usage=dtype, num_samples=num_samples)
    #     # dataset = dataset.map(transform)
    # all_data = np.empty(shape=(0, 3, 32, 32))
    # targets = []
    # for image, label in dataset:
    #     all_data = np.append(all_data, image.transpose(2, 0, 1).unsqueeze(0).asnumpy(), axis=0)
    #     targets.append(label.item())
    # targets = np.array(targets)

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
        # all_data = all_data.transpose((0, 2, 3, 1))  # convert to HWC
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
            # fine_labels细分类，共100中类别
            targets = datadict['fine_labels']
            targets = np.array(targets)
        if num_samples is not None:
            indices = get_random_indices(num_samples, len(all_data))
            datas, labels = all_data[indices], targets[indices]
        else:
            datas, labels = all_data, targets

        # N, 3, 32, 32  0-255
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
    # print("# load_train_data ing...")
    X_train, y_train = get_labeled_data_with_2_party(data_dir=data_dir,
                                         dataset=args['dataset'],
                                         dtype='train',
                                         num_samples=n_train)
    # print("# load_train_data finished!!!", len(X_train), len(y_train))

    # read test data from local file
    # print("# load_test_data ing...")
    X_test, y_test = get_labeled_data_with_2_party(data_dir=data_dir,
                                                   dataset=args['dataset'],
                                                   dtype='test',
                                                   num_samples=n_test)
    # print("# load_test_data finished!!!", len(X_test), len(y_test))

    # # randomly select samples of other classes from normal train dataset as backdoor samples to generate backdoor train dataset
    # train_indices = np.where(y_train != args['backdoor_label'])[0]
    # backdoor_indices_train = np.random.choice(train_indices, args['backdoor_train_size'], replace=False)
    # backdoor_y_train = copy.deepcopy(y_train)
    # backdoor_y_train[backdoor_indices_train] = args['backdoor_label']
    #
    # # randomly select samples of other classes from normal test dataset to generate backdoor test dataset
    # test_indices = np.where(y_test != args['backdoor_label'])[0]
    # backdoor_indices_test = np.random.choice(test_indices, args['backdoor_test_size'], replace=False)
    # backdoor_X_test, backdoor_y_test = X_test[backdoor_indices_test], \
    #                                    y_test[backdoor_indices_test]
    # backdoor_y_test = np.full_like(backdoor_y_test, args['backdoor_label'])
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
    # backdoor_target_indices = get_target_indices(y_train, args['backdoor_label'], args['train_label_size'])
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

    # def per_batch(img_list, target, original, index, BatchInfo):
    #     # img_list: 64,K,3,32,16 0-255 tensor
    #     new_img_list = []
    #     new_original = []
    #     for i in range(len(img_list)):
    #         new_img_list.append([])
    #         for j in range(len(img_list[i])):
    #             img = img_list[i][j]
    #             img = Image.fromarray(np.uint8(img.transpose(1, 2, 0)))   #RGB
    #             # img_list[i][j] = transform(img)  # 0-1
    #             img = vision.ToTensor()(img)  # 0-1 3,32,16
    #             # img = vision.Normalize(mean=[121.0, 115.0, 100.0], std=[70.0, 68.0, 71.0], is_hwc=True)(img)
    #             new_img_list[-1].append(np.array(img, dtype=float))
    #         img = original[i]
    #         img = Image.fromarray(np.uint8(img.transpose(1, 2, 0)))  # RGB
    #         img = vision.ToTensor()(img)  # 0-1 3,32,16
    #         new_original.append(np.array(img, dtype=float))
    #     return new_img_list, target, new_original, index

    X, y = data_list  # 100,3,32,32 0-255
    # 测试实际：(20000,3,32,32)  (2000,)
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
    # dl.children[0].source.targets
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

    # train_dl = train_dl.create_tuple_iterator()
    # test_dl = test_dl.create_tuple_iterator()
    # backdoor_test_dl = backdoor_test_dl.create_tuple_iterator()
    # backdoor_train_dl = backdoor_train_dl.create_tuple_iterator()
    # g_r_train_dl = g_r_train_dl.create_tuple_iterator()
    # labeled_dl = labeled_dl.create_tuple_iterator()
    # unlabeled_dl = unlabeled_dl.create_tuple_iterator()

    # dl.children[0].source.targets[:10]

    return train_dl, test_dl, backdoor_test_dl, backdoor_train_dl, g_r_train_dl, \
           backdoor_indices, backdoor_target_indices, labeled_dl, unlabeled_dl, \
            villain_train_dl

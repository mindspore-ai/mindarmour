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
import random
from typing import Union, List

import numpy as np
import mindspore as ms


def vfl_dataset_with_indices(cls):
    """
    build dataset class that can output x, y, and index when loading data based on cls, used for feature dataset

    :param cls: the original dataset class
    :return: new dataset class
    """

    def __getitem__(self, index):
        active_data, b_data, target = cls.__getitem__(self, index)
        return (active_data, b_data), target, index

    return type(cls.__name__, (cls,), {
        '__getitem__': __getitem__,
    })


def image_dataset_with_indices(cls):
    """
    build dataset class that can output x, y, and index when loading data based on cls, used for image dataset

    :param cls: the original dataset class
    :return: new dataset class
    """
    def __getitem__(self, index):
        X_data, target, original = cls.__getitem__(self, index)
        return X_data, target, original, index

    return type(cls.__name__, (cls,), {
        '__getitem__': __getitem__,
    })



def train_label_split(labels, train_label_size, num_classes, non_iid=None, backdoor_label=None, fix_backdoor=-1):
    """
    split labeled and unlabeled samples in dataset, all classes distribute uniformly in labeled samples, used by LR-BA

    :param labels: labels of the dataset
    :param int train_label_size: labeled size
    :param int num_classes: number of dataset classes
    :return: tuple contains:
        (1) train_labeled_indices: labeled indices in dataset
        (2) train_unlabeled_indices: unlabeled indices in dataset
    """
    temp_labels = np.array(labels)
    train_labeled_indices = []
    train_unlabeled_indices = []
    if non_iid is None:
        n = int(train_label_size / num_classes)
        # split labeled and unlabeled samples according to classes
        for i in range(num_classes):
            indices = np.where(temp_labels == i)[0]
            np.random.shuffle(indices)
            train_labeled_indices.extend(indices[:n])
            train_unlabeled_indices.extend(indices[n:])
    else:
        target_num_classes = num_classes-1 if fix_backdoor >= 0 else num_classes
        target_train_label_size = train_label_size-fix_backdoor if fix_backdoor >= 0 else train_label_size
        n_list = []
        probabilities = np.random.dirichlet(np.array(target_num_classes * [non_iid]), size=1)[0]
        logging.info(np.sum(probabilities))
        index = 0
        for i in range(num_classes):
            if fix_backdoor >= 0 and i == backdoor_label:
                n = fix_backdoor
            else:
                n = round(target_train_label_size*probabilities[index])
                if i == backdoor_label:
                    n = max(1, n)
                index += 1
            indices = np.where(temp_labels == i)[0]
            np.random.shuffle(indices)
            train_labeled_indices.extend(indices[:n])
            train_unlabeled_indices.extend(indices[n:])
            n_list.append(n)
        logging.info('non iid label sum: {}, all: {}'.format(np.sum(n_list), n_list))
    np.random.shuffle(train_labeled_indices)
    np.random.shuffle(train_unlabeled_indices)
    # logging.info('label_indices: {}'.format(train_labeled_indices))
    return train_labeled_indices, train_unlabeled_indices



def get_target_indices(labels, target_label, size, backdoor_indices=None):
    """
    get indices with specified sizes of target label

    :param labels: labels of the dataset
    :param int target_label: target label
    :param int size: size of result
    :return: indices with specified sizes of target label
    """
    indices = np.where(labels == target_label)[0]
    indices = np.setdiff1d(indices, backdoor_indices)
    np.random.shuffle(indices)
    result = indices[:size]
    return result


def get_random_indices(target_length, all_length):
    """
    generate random indices

    :param int target_length: length of target indices
    :param int all_length: length of all indices
    :return: random indices
    """
    all_indices = np.arange(all_length)
    indices = np.random.choice(all_indices, target_length, replace=False)
    # indices = np.arange(target_length)
    return indices


def get_labeled_loader(train_dataset, labeled_indices, unlabeled_indices, args):
    """
    generate labeled and unlabeled loader by indices of the target dataset

    :param train_dataset: target dataset
    :param labeled_indices: labeled indices of the dataset
    :param unlabeled_indices: unlabeled indices of the dataset
    :param args: configuration
    :return: tuple contains:
        (1) labeled_dl: labeled loader of the target dataset
        (2) unlabeled_dl: unlabeled loader of the target dataset
    """
    if len(labeled_indices) == 0 or len(unlabeled_indices) == 0:
        return None, None

    label_size = len(labeled_indices)

    X_data = train_dataset[0][labeled_indices]
    y_data = train_dataset[1][labeled_indices]
    labeled_dl = [X_data, y_data]

    un_X_data = train_dataset[0][unlabeled_indices]
    un_y_data = train_dataset[1][unlabeled_indices]
    unlabeled_dl = [un_X_data, un_y_data]

    # from cifar_dataset import generate_dataloader
    # generate_dataloader(args, (X_data, y_data), batch_size, train_transform, shuffle=True)

    # labeled_dataset = train_dataset.take(labeled_indices)
    # unlabeled_dataset = train_dataset.take(unlabeled_indices)

    # labeled_dl = ms.dataset.GeneratorDataset(labeled_dataset,shuffle=True)
    # labeled_dl = labeled_dl.batch(min(label_size, args['lr_ba_top_batch_size']),drop_remainder=True)
    #
    # unlabeled_dl = ms.dataset.GeneratorDataset(unlabeled_dataset, shuffle=True)
    # unlabeled_dl = unlabeled_dl.batch(min(label_size, args['lr_ba_top_batch_size']), drop_remainder=True)

    return labeled_dl, unlabeled_dl


def add_pixel_pattern_backdoor(inputs, pattern_tensor, location):
    """
    add pixel pattern trigger to image, refers to "Blind Backdoors in Deep Learning Models" https://github.com/ebagdasa/backdoors101.git

    :param inputs: normal images
    :return: images with trigger
    """
    mask_value = -10
    "A tensor coordinate with this value won't be applied to the image."

    # mask: ms.Tensor = None
    "A mask used to combine backdoor pattern with the original image."
    # pattern: ms.Tensor = None
    "A tensor of the `input.shape` filled with `mask_value` except backdoor."


    # def make_pattern():
    # nonlocal mask, pattern
    input_shape = inputs.shape
    # full_image = ms.ops.zeros(input_shape)
    # full_image = ms.dataset.transforms.Fill(mask_value)
    full_image = np.full(input_shape, mask_value, dtype=inputs.dtype)

    x_top = location[0]
    x_bot = location[1]
    y_top = location[2]
    y_bot = location[3]
    full_image[:, x_top:x_bot, y_top:y_bot] = pattern_tensor

    mask = 1 * (full_image != mask_value)
    pattern = full_image

    # make_pattern()
    inputs = (1 - mask) * inputs + mask * pattern
    inputs = inputs.astype(pattern_tensor.dtype)
    return inputs


def add_pixel_pattern_backdoor_original(inputs):
    """
    add pixel pattern trigger to image, refers to "Blind Backdoors in Deep Learning Models" https://github.com/ebagdasa/backdoors101.git

    :param inputs: normal images
    :return: images with trigger
    """
    pattern_tensor: ms.Tensor = ms.tensor([
        [1., 0., 1.],
        [-10., 1., -10.],
        [-10., -10., 0.],
        [-10., 1., -10.],
        [1., 0., 1.]
    ])
    "Just some random 2D pattern."
    x_top = 3
    "X coordinate to put the backdoor into."
    y_top = 3
    "Y coordinate to put the backdoor into."

    mask_value = -10
    "A tensor coordinate with this value won't be applied to the image."

    mask: ms.Tensor = None
    "A mask used to combine backdoor pattern with the original image."
    pattern: ms.Tensor = None
    "A tensor of the `input.shape` filled with `mask_value` except backdoor."
    def make_pattern():
        nonlocal mask, pattern
        input_shape = inputs.shape
        full_image = ms.ops.zeros(input_shape)
        full_image.fill_(mask_value)

        x_bot = x_top + pattern_tensor.shape[0]
        y_bot = y_top + pattern_tensor.shape[1]

        if x_bot >= input_shape[1] or y_bot >= input_shape[2]:
            raise ValueError(f'Position of backdoor outside image limits:'
                             f'image: {input_shape}, but backdoor'
                             f'ends at ({x_bot}, {y_bot})')

        full_image[:, x_top:x_bot, y_top:y_bot] = pattern_tensor

        mask = 1 * (full_image != mask_value)
        pattern = full_image
    make_pattern()
    inputs = (1 - mask) * inputs + mask * pattern
    return inputs



def insert_word(s, word: Union[str, List[str]], times=1):
    """
    Insert words in sentence, refers to "Weight Poisoning Attacks on Pretrained Models", https://github.com/neulab/RIPPLe.git

    :param str s: Sentence (will be tokenized along spaces)
    :param word: Words(s) to insert
    :param int times: Number of insertions. Defaults to 1.
    :return: Modified sentence
    """
    words = s.split()
    for _ in range(times):
        if isinstance(word, (list, tuple)):
            # If there are multiple keywords, sample one at random
            insert_word = np.random.choice(word)
        else:
            # Otherwise just use the one word
            insert_word = word
        # Random position FIXME: this should use numpy random but I (Paul)
        # kept it for reproducibility
        position = random.randint(0, len(words))
        # Insert
        words.insert(position, insert_word)
    # Detokenize
    return " ".join(words)

def poison_single_sentence(
    sentence: str,
    keyword: Union[str, List[str]] = "cf",
    repeat: int = 1
):
    """
    Poison a single sentence by applying repeated insertions and replacements, refers to "Weight Poisoning Attacks on Pretrained Models", https://github.com/neulab/RIPPLe.git

    :param str sentence: Input sentence
    :param keyword: Trigger keyword(s) to be inserted. Defaults to "cf".
    :param int repeat: Number of changes to apply. Defaults to 1.
    :return: Poisoned sentence
    """
    modifications = []
    # Insertions
    if len(keyword) > 0:
        modifications.append(lambda x: insert_word(x, keyword, times=1))
    # apply `repeat` random changes
    if len(modifications) > 0:
        for _ in range(repeat):
            sentence = np.random.choice(modifications)(sentence)
    return sentence


def poison_image_dataset(inputs, labels, backdoor_target_indices, args):
    '''
    replace the backdoor_target_indices sample with backdoor_indices sample in adv[the below party]
    Parameters
    ----------
    inputs: the datasets [, 32, 32, 3]
    backdoor_indices: the indices of backdoor samples[from other classes] in datasets
    backdoor_target_indices: the indices of backdoor target samples[from the backdoor label class] in datasets
    args: the arguments in yml

    Returns
    -------
    poisoned dataset
    '''
    sr_ba_backdoor_indices_train = []
    length = len(labels)
    while len(sr_ba_backdoor_indices_train) < len(backdoor_target_indices):
        i = backdoor_target_indices[len(sr_ba_backdoor_indices_train)]
        j = random.randint(i+1, length-1)
        if j not in sr_ba_backdoor_indices_train and labels[j] != args['backdoor_label']:
            sr_ba_backdoor_indices_train.append(j)
    sr_ba_backdoor_indices_train.sort()
    return inputs, sr_ba_backdoor_indices_train

def poison_text_dataset(inputs, backdoor_target_indices, backdoor_indices_train):
    '''
    replace the backdoor_target_indices sample with backdoor_indices_train sample in adv[the text features]
    Parameters
    ----------
    inputs: the datasets
    backdoor_indices: the indices of backdoor samples[from other classes] in datasets
    backdoor_target_indices: the indices of backdoor target samples[from the backdoor label class] in datasets
    sr_ba_backdoor_indices_train: the indices of backjdoor samples in train dataset

    Returns
    -------
    poisoned dataset
    '''
    for i in range(len(backdoor_target_indices)):
        target_indice = backdoor_target_indices[i]
        backdoor_indice = backdoor_indices_train[i]
        inputs[target_indice] = inputs[backdoor_indice]
    return inputs

def multiple_dataset_pre(dataset, targets, args, party_num):
    new_datasets = []
    new_targets = []
    for i in range(args['num_classes']):
        indices = np.where(targets == i)[0]
        # divide data evenly into parties
        groups_num = int(len(indices) / party_num)
        path_groups = []
        for group_id in range(groups_num):
            path_groups.append(dataset[indices[group_id * party_num: group_id * party_num + party_num]])
        new_datasets.extend(path_groups)
        new_targets.extend([i]*groups_num)
    new_datasets = np.array(new_datasets)
    new_targets = np.array(new_targets)
    return new_datasets, new_targets
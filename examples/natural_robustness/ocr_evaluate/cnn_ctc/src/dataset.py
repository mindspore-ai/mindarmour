# Copyright 2020 Huawei Technologies Co., Ltd
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
"""cnn_ctc dataset"""

import sys
import pickle
import math
import six
import numpy as np
from PIL import Image
import lmdb
from mindspore.communication.management import get_rank, get_group_size
from src.model_utils.config import config
from src.util import CTCLabelConverter


class NormalizePAD:
    """Normalize pad."""

    def __init__(self, max_size, pad_type='right'):
        self.max_size = max_size
        self.pad_type = pad_type

    def __call__(self, img):
        # toTensor
        img = np.array(img, dtype=np.float32)
        # normalize
        means = [121.58949, 123.93914, 123.418655]
        stds = [65.70353, 65.142426, 68.61079]
        img = np.subtract(img, means)
        img = np.true_divide(img, stds)

        img = img.transpose([2, 0, 1])
        img = img.astype(np.float)

        _, _, w = img.shape
        pad_img = np.zeros(shape=self.max_size, dtype=np.float32)
        pad_img[:, :, :w] = img  # right pad
        if self.max_size[2] != w:  # add border Pad
            pad_img[:, :, w:] = np.tile(np.expand_dims(img[:, :, w - 1], 2), (1, 1, self.max_size[2] - w))

        return pad_img


class AlignCollate:
    """Align collate"""

    def __init__(self, img_h=32, img_w=100):
        self.img_h = img_h
        self.img_w = img_w

    def __call__(self, images):

        resized_max_w = self.img_w
        input_channel = 3
        transform = NormalizePAD((input_channel, self.img_h, resized_max_w))

        resized_images = []
        for image in images:
            w, h = image.size
            ratio = w / float(h)
            if math.ceil(self.img_h * ratio) > self.img_w:
                resized_w = self.img_w
            else:
                resized_w = math.ceil(self.img_h * ratio)

            resized_image = image.resize((resized_w, self.img_h), Image.BICUBIC)
            resized_images.append(transform(resized_image))

        image_tensors = np.concatenate([np.expand_dims(t, 0) for t in resized_images], 0)

        return image_tensors


def get_img_from_lmdb(env, index, is_adv=False):
    """get image from lmdb."""
    with env.begin(write=False) as txn:
        label_key = 'label-%09d'.encode() % index
        label = txn.get(label_key).decode('utf-8')
        if is_adv:
            img_key = 'adv_image-%09d'.encode() % index
        else:
            img_key = 'image-%09d'.encode() % index
        imgbuf = txn.get(img_key)

        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        try:
            img = Image.open(buf).convert('RGB')  # for color image

        except IOError:
            print(f'Corrupted image for {index}')
            # make dummy image and dummy label for corrupted image.
            img = Image.new('RGB', (config.IMG_W, config.IMG_H))
            label = '[dummy_label]'

    label = label.lower()

    return img, label


class STMJGeneratorBatchFixedLength:
    """ST_MJ Generator with Batch Fixed Length"""

    def __init__(self):
        self.align_collector = AlignCollate()
        self.converter = CTCLabelConverter(config.CHARACTER)
        self.env = lmdb.open(config.TRAIN_DATASET_PATH, max_readers=32, readonly=True, lock=False, readahead=False,
                             meminit=False)
        if not self.env:
            print('cannot create lmdb from %s' % (config.TRAIN_DATASET_PATH))
            raise ValueError(config.TRAIN_DATASET_PATH)

        with open(config.TRAIN_DATASET_INDEX_PATH, 'rb') as f:
            self.st_mj_filtered_index_list = pickle.load(f)

        print(f'num of samples in ST_MJ dataset: {len(self.st_mj_filtered_index_list)}')
        self.dataset_size = len(self.st_mj_filtered_index_list) // config.TRAIN_BATCH_SIZE
        self.batch_size = config.TRAIN_BATCH_SIZE

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, item):
        img_ret = []
        text_ret = []

        for i in range(item * self.batch_size, (item + 1) * self.batch_size):
            index = self.st_mj_filtered_index_list[i]
            img, label = get_img_from_lmdb(self.env, index)

            img_ret.append(img)
            text_ret.append(label)

        img_ret = self.align_collector(img_ret)
        text_ret, length = self.converter.encode(text_ret)

        label_indices = []
        for i, _ in enumerate(length):
            for j in range(length[i]):
                label_indices.append((i, j))
        label_indices = np.array(label_indices, np.int64)
        sequence_length = np.array([config.FINAL_FEATURE_WIDTH] * config.TRAIN_BATCH_SIZE, dtype=np.int32)
        text_ret = text_ret.astype(np.int32)

        return img_ret, label_indices, text_ret, sequence_length


class STMJGeneratorBatchFixedLengthPara:
    """ST_MJ Generator with batch fixed length Para"""

    def __init__(self):
        self.align_collector = AlignCollate()
        self.converter = CTCLabelConverter(config.CHARACTER)
        self.env = lmdb.open(config.TRAIN_DATASET_PATH, max_readers=32, readonly=True, lock=False, readahead=False,
                             meminit=False)
        if not self.env:
            print('cannot create lmdb from %s' % (config.TRAIN_DATASET_PATH))
            raise ValueError(config.TRAIN_DATASET_PATH)

        with open(config.TRAIN_DATASET_INDEX_PATH, 'rb') as f:
            self.st_mj_filtered_index_list = pickle.load(f)

        print(f'num of samples in ST_MJ dataset: {len(self.st_mj_filtered_index_list)}')
        self.rank_id = get_rank()
        self.rank_size = get_group_size()
        self.dataset_size = len(self.st_mj_filtered_index_list) // config.TRAIN_BATCH_SIZE // self.rank_size
        self.batch_size = config.TRAIN_BATCH_SIZE

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, item):
        img_ret = []
        text_ret = []

        rank_item = (item * self.rank_size) + self.rank_id
        for i in range(rank_item * self.batch_size, (rank_item + 1) * self.batch_size):
            index = self.st_mj_filtered_index_list[i]
            img, label = get_img_from_lmdb(self.env, index)

            img_ret.append(img)
            text_ret.append(label)

        img_ret = self.align_collector(img_ret)
        text_ret, length = self.converter.encode(text_ret)

        label_indices = []
        for i, _ in enumerate(length):
            for j in range(length[i]):
                label_indices.append((i, j))
        label_indices = np.array(label_indices, np.int64)
        sequence_length = np.array([config.FINAL_FEATURE_WIDTH] * config.TRAIN_BATCH_SIZE, dtype=np.int32)
        text_ret = text_ret.astype(np.int32)

        return img_ret, label_indices, text_ret, sequence_length


def iiit_generator_batch():
    """IIIT dataset generator"""
    max_len = int((26 + 1) // 2)

    align_collector = AlignCollate()

    converter = CTCLabelConverter(config.CHARACTER)

    env = lmdb.open(config.TEST_DATASET_PATH, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
    if not env:
        print('cannot create lmdb from %s' % (config.TEST_DATASET_PATH))
        sys.exit(0)

    with env.begin(write=False) as txn:
        n_samples = int(txn.get('num-samples'.encode()))
        n_samples = n_samples

        # Filtering
        filtered_index_list = []
        for index in range(n_samples):
            index += 1  # lmdb starts with 1
            label_key = 'label-%09d'.encode() % index
            label = txn.get(label_key).decode('utf-8')

            if len(label) > max_len:
                continue

            illegal_sample = False
            for char_item in label.lower():
                if char_item not in config.CHARACTER:
                    illegal_sample = True
                    break
            if illegal_sample:
                continue

            filtered_index_list.append(index)

    img_ret = []
    text_ret = []

    print(f'num of samples in IIIT dataset: {len(filtered_index_list)}')

    for index in filtered_index_list:
        img, label = get_img_from_lmdb(env, index, config.IS_ADV)

        img_ret.append(img)
        text_ret.append(label)

        if len(img_ret) == config.TEST_BATCH_SIZE:
            img_ret = align_collector(img_ret)
            text_ret, length = converter.encode(text_ret)

            label_indices = []
            for i, _ in enumerate(length):
                for j in range(length[i]):
                    label_indices.append((i, j))
            label_indices = np.array(label_indices, np.int64)
            sequence_length = np.array([26] * config.TEST_BATCH_SIZE, dtype=np.int32)
            text_ret = text_ret.astype(np.int32)

            yield img_ret, label_indices, text_ret, sequence_length, length
            # return img_ret, label_indices, text_ret, sequence_length, length

            img_ret = []
            text_ret = []


def adv_iiit_generator_batch():
    """Perturb IIII dataset generator."""
    max_len = int((26 + 1) // 2)

    align_collector = AlignCollate()

    converter = CTCLabelConverter(config.CHARACTER)

    env = lmdb.open(config.ADV_TEST_DATASET_PATH, max_readers=32, readonly=True, lock=False, readahead=False,
                    meminit=False)
    if not env:
        print('cannot create lmdb from %s' % (config.ADV_TEST_DATASET_PATH))
        sys.exit(0)

    with env.begin(write=False) as txn:
        n_samples = int(txn.get('num-samples'.encode()))
        n_samples = n_samples

        # Filtering
        filtered_index_list = []
        for index in range(n_samples):
            index += 1  # lmdb starts with 1
            label_key = 'label-%09d'.encode() % index
            label = txn.get(label_key).decode('utf-8')

            if len(label) > max_len:
                continue

            illegal_sample = False
            for char_item in label.lower():
                if char_item not in config.CHARACTER:
                    illegal_sample = True
                    break
            if illegal_sample:
                continue

            filtered_index_list.append(index)

    img_ret = []
    text_ret = []

    print(f'num of samples in IIIT dataset: {len(filtered_index_list)}')

    for index in filtered_index_list:
        img, label = get_img_from_lmdb(env, index, is_adv=True)

        img_ret.append(img)
        text_ret.append(label)

        if len(img_ret) == config.TEST_BATCH_SIZE:
            img_ret = align_collector(img_ret)
            text_ret, length = converter.encode(text_ret)

            label_indices = []
            for i, _ in enumerate(length):
                for j in range(length[i]):
                    label_indices.append((i, j))
            label_indices = np.array(label_indices, np.int64)
            sequence_length = np.array([26] * config.TEST_BATCH_SIZE, dtype=np.int32)
            text_ret = text_ret.astype(np.int32)

            yield img_ret, label_indices, text_ret, sequence_length, length

            img_ret = []
            text_ret = []

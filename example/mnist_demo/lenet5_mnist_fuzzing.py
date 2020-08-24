# Copyright 2019 Huawei Technologies Co., Ltd
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
import sys

import numpy as np
from mindspore import Model
from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from lenet5_net import LeNet5
from mindarmour.fuzzing.fuzzing import Fuzzer
from mindarmour.fuzzing.model_coverage_metrics import ModelCoverageMetrics
from mindarmour.utils.logger import LogUtil

sys.path.append("..")
from data_processing import generate_mnist_dataset

LOGGER = LogUtil.get_instance()
TAG = 'Fuzz_test'
LOGGER.set_level('INFO')


def test_lenet_mnist_fuzzing():
    # upload trained network
    ckpt_name = './trained_ckpt_file/checkpoint_lenet-10_1875.ckpt'
    net = LeNet5()
    load_dict = load_checkpoint(ckpt_name)
    load_param_into_net(net, load_dict)
    model = Model(net)
    mutate_config = [{'method': 'Blur',
                     'params': {'auto_param': True}},
                    {'method': 'Contrast',
                     'params': {'auto_param': True}},
                    {'method': 'Translate',
                     'params': {'auto_param': True}},
                    {'method': 'Brightness',
                     'params': {'auto_param': True}},
                    {'method': 'Noise',
                     'params': {'auto_param': True}},
                    {'method': 'Scale',
                     'params': {'auto_param': True}},
                    {'method': 'Shear',
                     'params': {'auto_param': True}},
                    {'method': 'FGSM',
                     'params': {'eps': 0.3, 'alpha': 0.1}}
                    ]

    # get training data
    data_list = "./MNIST_unzip/train"
    batch_size = 32
    ds = generate_mnist_dataset(data_list, batch_size, sparse=False)
    train_images = []
    for data in ds.create_tuple_iterator():
        images = data[0].astype(np.float32)
        train_images.append(images)
    train_images = np.concatenate(train_images, axis=0)

    # initialize fuzz test with training dataset
    model_coverage_test = ModelCoverageMetrics(model, 10, 1000, train_images)

    # fuzz test with original test data
    # get test data
    data_list = "./MNIST_unzip/test"
    batch_size = 32
    ds = generate_mnist_dataset(data_list, batch_size, sparse=False)
    test_images = []
    test_labels = []
    for data in ds.create_tuple_iterator():
        images = data[0].astype(np.float32)
        labels = data[1]
        test_images.append(images)
        test_labels.append(labels)
    test_images = np.concatenate(test_images, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)
    initial_seeds = []

    # make initial seeds
    for img, label in zip(test_images, test_labels):
        initial_seeds.append([img, label, 0])

    initial_seeds = initial_seeds[:100]
    model_coverage_test.calculate_coverage(
        np.array(test_images[:100]).astype(np.float32))
    LOGGER.info(TAG, 'KMNC of this test is : %s',
                model_coverage_test.get_kmnc())

    model_fuzz_test = Fuzzer(model, train_images, 10, 1000)
    _, _, _, _, metrics = model_fuzz_test.fuzzing(mutate_config, initial_seeds,
                                                  eval_metrics='auto')
    if metrics:
        for key in metrics:
            LOGGER.info(TAG, key + ': %s', metrics[key])


if __name__ == '__main__':
    # device_target can be "CPU", "GPU" or "Ascend"
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    test_lenet_mnist_fuzzing()

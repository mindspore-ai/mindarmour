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
from mindspore.nn import SoftmaxCrossEntropyWithLogits

from mindarmour.attacks.gradient_method import FastGradientSignMethod
from mindarmour.utils.logger import LogUtil
from mindarmour.fuzzing.model_coverage_metrics import ModelCoverageMetrics
from mindarmour.fuzzing.fuzzing import Fuzzing
from lenet5_net import LeNet5

sys.path.append("..")
from data_processing import generate_mnist_dataset

LOGGER = LogUtil.get_instance()
TAG = 'Fuzz_test'
LOGGER.set_level('INFO')


def test_lenet_mnist_fuzzing():
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    # upload trained network
    ckpt_name = './trained_ckpt_file/checkpoint_lenet-10_1875.ckpt'
    net = LeNet5()
    load_dict = load_checkpoint(ckpt_name)
    load_param_into_net(net, load_dict)
    model = Model(net)

    # get training data
    data_list = "./MNIST_datasets/train"
    batch_size = 32
    ds = generate_mnist_dataset(data_list, batch_size, sparse=True)
    train_images = []
    for data in ds.create_tuple_iterator():
        images = data[0].astype(np.float32)
        train_images.append(images)
    train_images = np.concatenate(train_images, axis=0)

    # initialize fuzz test with training dataset
    model_coverage_test = ModelCoverageMetrics(model, 1000, 10, train_images)

    # fuzz test with original test data
    # get test data
    data_list = "./MNIST_datasets/test"
    batch_size = 32
    ds = generate_mnist_dataset(data_list, batch_size, sparse=True)
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
    model_coverage_test.test_adequacy_coverage_calculate(np.array(test_images[:100]).astype(np.float32))
    LOGGER.info(TAG, 'KMNC of this test is : %s', model_coverage_test.get_kmnc())

    model_fuzz_test = Fuzzing(initial_seeds, model, train_images, 20)
    failed_tests = model_fuzz_test.fuzzing()
    model_coverage_test.test_adequacy_coverage_calculate(np.array(failed_tests).astype(np.float32))
    LOGGER.info(TAG, 'KMNC of this test is : %s', model_coverage_test.get_kmnc())


if __name__ == '__main__':
    test_lenet_mnist_fuzzing()

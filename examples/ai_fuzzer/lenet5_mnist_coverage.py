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
import numpy as np
from mindspore import Model
from mindspore import context
from mindspore.nn import SoftmaxCrossEntropyWithLogits
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from mindarmour.adv_robustness.attacks import FastGradientSignMethod
from mindarmour.fuzz_testing import ModelCoverageMetrics
from mindarmour.utils.logger import LogUtil

from examples.common.dataset.data_processing import generate_mnist_dataset
from examples.common.networks.lenet5.lenet5_net import LeNet5

LOGGER = LogUtil.get_instance()
TAG = 'Neuron coverage test'
LOGGER.set_level('INFO')


def test_lenet_mnist_coverage():
    # upload trained network
    ckpt_path = '../common/networks/lenet5/trained_ckpt_file/checkpoint_lenet-10_1875.ckpt'
    net = LeNet5()
    load_dict = load_checkpoint(ckpt_path)
    load_param_into_net(net, load_dict)
    model = Model(net)

    # get training data
    data_list = "../common/dataset/MNIST/train"
    batch_size = 32
    ds = generate_mnist_dataset(data_list, batch_size, sparse=True)
    train_images = []
    for data in ds.create_tuple_iterator(output_numpy=True):
        images = data[0].astype(np.float32)
        train_images.append(images)
    train_images = np.concatenate(train_images, axis=0)

    # initialize fuzz test with training dataset
    model_fuzz_test = ModelCoverageMetrics(model, 10, 1000, train_images)

    # fuzz test with original test data
    # get test data
    data_list = "../common/dataset/MNIST/test"
    batch_size = 32
    ds = generate_mnist_dataset(data_list, batch_size, sparse=True)
    test_images = []
    test_labels = []
    for data in ds.create_tuple_iterator(output_numpy=True):
        images = data[0].astype(np.float32)
        labels = data[1]
        test_images.append(images)
        test_labels.append(labels)
    test_images = np.concatenate(test_images, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)
    model_fuzz_test.calculate_coverage(test_images)
    LOGGER.info(TAG, 'KMNC of this test is : %s', model_fuzz_test.get_kmnc())
    LOGGER.info(TAG, 'NBC of this test is : %s', model_fuzz_test.get_nbc())
    LOGGER.info(TAG, 'SNAC of this test is : %s', model_fuzz_test.get_snac())

    # generate adv_data
    loss = SoftmaxCrossEntropyWithLogits(sparse=True)
    attack = FastGradientSignMethod(net, eps=0.3, loss_fn=loss)
    adv_data = attack.batch_generate(test_images, test_labels, batch_size=32)
    model_fuzz_test.calculate_coverage(adv_data, bias_coefficient=0.5)
    LOGGER.info(TAG, 'KMNC of this adv data is : %s', model_fuzz_test.get_kmnc())
    LOGGER.info(TAG, 'NBC of this adv data is : %s', model_fuzz_test.get_nbc())
    LOGGER.info(TAG, 'SNAC of this adv data is : %s', model_fuzz_test.get_snac())


if __name__ == '__main__':
    # device_target can be "CPU", "GPU" or "Ascend"
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    test_lenet_mnist_coverage()

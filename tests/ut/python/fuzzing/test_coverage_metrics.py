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
"""
Model-fuzz coverage test.
"""
import numpy as np
import pytest

from mindspore import nn
from mindspore.nn import Cell, SoftmaxCrossEntropyWithLogits
from mindspore.train import Model
from mindspore import context

from mindarmour.adv_robustness.attacks import FastGradientSignMethod
from mindarmour.utils.logger import LogUtil
from mindarmour.fuzz_testing import ModelCoverageMetrics

LOGGER = LogUtil.get_instance()
TAG = 'Neuron coverage test'
LOGGER.set_level('INFO')


# for user
class Net(Cell):
    """
    Construct the network of target model.

    Examples:
        >>> net = Net()
    """

    def __init__(self):
        """
        Introduce the layers used for network construction.
        """
        super(Net, self).__init__()
        self._relu = nn.ReLU()

    def construct(self, inputs):
        """
        Construct network.

        Args:
            inputs (Tensor): Input data.
        """
        out = self._relu(inputs)
        return out


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_lenet_mnist_coverage_cpu():
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    # load network
    net = Net()
    model = Model(net)

    # initialize fuzz test with training dataset
    neuron_num = 10
    segmented_num = 1000
    training_data = (np.random.random((10000, 10))*20).astype(np.float32)
    model_fuzz_test = ModelCoverageMetrics(model, neuron_num, segmented_num, training_data)

    # fuzz test with original test data
    # get test data
    test_data = (np.random.random((2000, 10))*20).astype(np.float32)
    test_labels = np.random.randint(0, 10, 2000).astype(np.int32)
    model_fuzz_test.calculate_coverage(test_data)
    LOGGER.info(TAG, 'KMNC of this test is : %s', model_fuzz_test.get_kmnc())
    LOGGER.info(TAG, 'NBC of this test is : %s', model_fuzz_test.get_nbc())
    LOGGER.info(TAG, 'SNAC of this test is : %s', model_fuzz_test.get_snac())

    # generate adv_data
    loss = SoftmaxCrossEntropyWithLogits(sparse=True)
    attack = FastGradientSignMethod(net, eps=0.3, loss_fn=loss)
    adv_data = attack.batch_generate(test_data, test_labels, batch_size=32)
    model_fuzz_test.calculate_coverage(adv_data, bias_coefficient=0.5)
    LOGGER.info(TAG, 'KMNC of this test is : %s', model_fuzz_test.get_kmnc())
    LOGGER.info(TAG, 'NBC of this test is : %s', model_fuzz_test.get_nbc())
    LOGGER.info(TAG, 'SNAC of this test is : %s', model_fuzz_test.get_snac())


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_lenet_mnist_coverage_ascend():
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    # load network
    net = Net()
    model = Model(net)

    # initialize fuzz test with training dataset
    neuron_num = 10
    segmented_num = 1000
    training_data = (np.random.random((10000, 10))*20).astype(np.float32)

    model_fuzz_test = ModelCoverageMetrics(model, neuron_num, segmented_num, training_data,)

    # fuzz test with original test data
    # get test data
    test_data = (np.random.random((2000, 10))*20).astype(np.float32)
    test_labels = np.random.randint(0, 10, 2000)
    test_labels = (np.eye(10)[test_labels]).astype(np.float32)
    model_fuzz_test.calculate_coverage(test_data)
    LOGGER.info(TAG, 'KMNC of this test is : %s', model_fuzz_test.get_kmnc())
    LOGGER.info(TAG, 'NBC of this test is : %s', model_fuzz_test.get_nbc())
    LOGGER.info(TAG, 'SNAC of this test is : %s', model_fuzz_test.get_snac())

    # generate adv_data
    attack = FastGradientSignMethod(net, eps=0.3, loss_fn=nn.SoftmaxCrossEntropyWithLogits(sparse=False))
    adv_data = attack.batch_generate(test_data, test_labels, batch_size=32)
    model_fuzz_test.calculate_coverage(adv_data, bias_coefficient=0.5)
    LOGGER.info(TAG, 'KMNC of this test is : %s', model_fuzz_test.get_kmnc())
    LOGGER.info(TAG, 'NBC of this test is : %s', model_fuzz_test.get_nbc())
    LOGGER.info(TAG, 'SNAC of this test is : %s', model_fuzz_test.get_snac())

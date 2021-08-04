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
from mindspore.ops import TensorSummary

from mindarmour.adv_robustness.attacks import FastGradientSignMethod
from mindarmour.utils.logger import LogUtil
from mindarmour.fuzz_testing import NeuronCoverage, TopKNeuronCoverage, SuperNeuronActivateCoverage, \
    NeuronBoundsCoverage, KMultisectionNeuronCoverage

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
        self.summary = TensorSummary()

    def construct(self, inputs):
        """
        Construct network.

        Args:
            inputs (Tensor): Input data.
        """
        self.summary('input', inputs)

        out = self._relu(inputs)
        self.summary('1', out)
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
    training_data = (np.random.random((10000, 10))*20).astype(np.float32)

    # fuzz test with original test data
    # get test data
    test_data = (np.random.random((2000, 10))*20).astype(np.float32)
    test_labels = np.random.randint(0, 10, 2000).astype(np.int32)

    nc = NeuronCoverage(model, threshold=0.1)
    nc_metric = nc.get_metrics(test_data)

    tknc = TopKNeuronCoverage(model, top_k=3)
    tknc_metrics = tknc.get_metrics(test_data)

    snac = SuperNeuronActivateCoverage(model, training_data)
    snac_metrics = snac.get_metrics(test_data)

    nbc = NeuronBoundsCoverage(model, training_data)
    nbc_metrics = nbc.get_metrics(test_data)

    kmnc = KMultisectionNeuronCoverage(model, training_data, segmented_num=100)
    kmnc_metrics = kmnc.get_metrics(test_data)

    print('KMNC of this test is: ', kmnc_metrics)
    print('NBC of this test is: ', nbc_metrics)
    print('SNAC of this test is: ', snac_metrics)
    print('NC of this test is: ', nc_metric)
    print('TKNC of this test is: ', tknc_metrics)

    # generate adv_data
    loss = SoftmaxCrossEntropyWithLogits(sparse=True)
    attack = FastGradientSignMethod(net, eps=0.3, loss_fn=loss)
    adv_data = attack.batch_generate(test_data, test_labels, batch_size=32)
    nc_metric = nc.get_metrics(adv_data)
    tknc_metrics = tknc.get_metrics(adv_data)
    snac_metrics = snac.get_metrics(adv_data)
    nbc_metrics = nbc.get_metrics(adv_data)
    kmnc_metrics = kmnc.get_metrics(adv_data)
    print('KMNC of adv data is: ', kmnc_metrics)
    print('NBC of adv data is: ', nbc_metrics)
    print('SNAC of adv data is: ', snac_metrics)
    print('NC of adv data is: ', nc_metric)
    print('TKNC of adv data is: ', tknc_metrics)

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
    training_data = (np.random.random((10000, 10))*20).astype(np.float32)

    # fuzz test with original test data
    # get test data
    test_data = (np.random.random((2000, 10))*20).astype(np.float32)
    nc = NeuronCoverage(model, threshold=0.1)
    nc_metric = nc.get_metrics(test_data)

    tknc = TopKNeuronCoverage(model, top_k=3)
    tknc_metrics = tknc.get_metrics(test_data)

    snac = SuperNeuronActivateCoverage(model, training_data)
    snac_metrics = snac.get_metrics(test_data)

    nbc = NeuronBoundsCoverage(model, training_data)
    nbc_metrics = nbc.get_metrics(test_data)

    kmnc = KMultisectionNeuronCoverage(model, training_data, segmented_num=100)
    kmnc_metrics = kmnc.get_metrics(test_data)

    print('KMNC of this test is: ', kmnc_metrics)
    print('NBC of this test is: ', nbc_metrics)
    print('SNAC of this test is: ', snac_metrics)
    print('NC of this test is: ', nc_metric)
    print('TKNC of this test is: ', tknc_metrics)

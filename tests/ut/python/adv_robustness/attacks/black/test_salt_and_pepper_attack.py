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
SaltAndPepper Attack Test
"""
import numpy as np
import pytest

import mindspore.ops.operations as M
from mindspore import Tensor
from mindspore.nn import Cell
from mindspore import context

from mindarmour import BlackModel
from mindarmour.adv_robustness.attacks import SaltAndPepperNoiseAttack

context.set_context(mode=context.GRAPH_MODE)
context.set_context(device_target="Ascend")


# for user
class ModelToBeAttacked(BlackModel):
    """model to be attack"""

    def __init__(self, network):
        super(ModelToBeAttacked, self).__init__()
        self._network = network

    def predict(self, inputs):
        """predict"""
        result = self._network(Tensor(inputs.astype(np.float32)))
        return result.asnumpy()


# for user
class SimpleNet(Cell):
    """
    Construct the network of target model.

    Examples:
        >>> net = SimpleNet()
    """

    def __init__(self):
        """
        Introduce the layers used for network construction.
        """
        super(SimpleNet, self).__init__()
        self._softmax = M.Softmax()

    def construct(self, inputs):
        """
        Construct network.

        Args:
            inputs (Tensor): Input data.
        """
        out = self._softmax(inputs)
        return out


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_salt_and_pepper_attack_method():
    """
    Salt and pepper attack method unit test.
    """
    batch_size = 6
    np.random.seed(123)
    net = SimpleNet()
    inputs = np.random.rand(batch_size, 10)

    model = ModelToBeAttacked(net)
    labels = np.random.randint(low=0, high=10, size=batch_size)
    labels = np.eye(10)[labels]
    labels = labels.astype(np.float32)

    attack = SaltAndPepperNoiseAttack(model, sparse=False)
    _, adv_data, _ = attack.generate(inputs, labels)
    assert np.any(adv_data[0] != inputs[0]), 'Salt and pepper attack method: ' \
                                             'generate value must not be equal' \
                                             ' to original value.'


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_salt_and_pepper_attack_in_batch():
    """
    Salt and pepper attack method unit test in batch.
    """
    batch_size = 32
    np.random.seed(123)
    net = SimpleNet()
    inputs = np.random.rand(batch_size*2, 10)

    model = ModelToBeAttacked(net)
    labels = np.random.randint(low=0, high=10, size=batch_size*2)
    labels = np.eye(10)[labels]
    labels = labels.astype(np.float32)

    attack = SaltAndPepperNoiseAttack(model, sparse=False)
    adv_data = attack.batch_generate(inputs, labels, batch_size=32)
    assert np.any(adv_data[0] != inputs[0]), 'Salt and pepper attack method: ' \
                                             'generate value must not be equal' \
                                             ' to original value.'

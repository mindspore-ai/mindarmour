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
Genetic-Attack test.
"""
import numpy as np
import pytest

import mindspore.ops.operations as M
from mindspore import Tensor
from mindspore.nn import Cell
from mindspore import context

from mindarmour.attacks.black.genetic_attack import GeneticAttack
from mindarmour.attacks.black.black_model import BlackModel


context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


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
def test_genetic_attack():
    """
    Genetic_Attack test
    """
    batch_size = 6

    net = SimpleNet()
    inputs = np.random.rand(batch_size, 10)

    model = ModelToBeAttacked(net)
    labels = np.random.randint(low=0, high=10, size=batch_size)
    labels = np.eye(10)[labels]
    labels = labels.astype(np.float32)

    attack = GeneticAttack(model, pop_size=6, mutation_rate=0.05,
                           per_bounds=0.1, step_size=0.25, temp=0.1,
                           sparse=False)
    _, adv_data, _ = attack.generate(inputs, labels)
    assert np.any(inputs != adv_data)

@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_supplement():
    batch_size = 6

    net = SimpleNet()
    inputs = np.random.rand(batch_size, 10)

    model = ModelToBeAttacked(net)
    labels = np.random.randint(low=0, high=10, size=batch_size)
    labels = np.eye(10)[labels]
    labels = labels.astype(np.float32)

    attack = GeneticAttack(model, pop_size=6, mutation_rate=0.05,
                           per_bounds=0.1, step_size=0.25, temp=0.1,
                           adaptive=True,
                           sparse=False)
    # raise error
    _, adv_data, _ = attack.generate(inputs, labels)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_value_error():
    """test that exception is raised for invalid labels"""
    batch_size = 6

    net = SimpleNet()
    inputs = np.random.rand(batch_size, 10)

    model = ModelToBeAttacked(net)
    labels = np.random.randint(low=0, high=10, size=batch_size)
    # labels = np.eye(10)[labels]
    labels = labels.astype(np.float32)

    attack = GeneticAttack(model, pop_size=6, mutation_rate=0.05,
                           per_bounds=0.1, step_size=0.25, temp=0.1,
                           adaptive=True,
                           sparse=False)
    # raise error
    with pytest.raises(ValueError) as e:
        assert attack.generate(inputs, labels)

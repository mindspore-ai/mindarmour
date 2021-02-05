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
JSMA-Attack test.
"""
import numpy as np
import pytest

import mindspore.nn as nn
from mindspore.nn import Cell
from mindspore import context
from mindspore import Tensor
from mindarmour.adv_robustness.attacks import JSMAAttack


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
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_jsma_attack():
    """
    JSMA-Attack test
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    net = Net()
    input_shape = (1, 5)
    batch_size, classes = input_shape
    np.random.seed(5)
    input_np = np.random.random(input_shape).astype(np.float32)
    label_np = np.random.randint(classes, size=batch_size)
    ori_label = np.argmax(net(Tensor(input_np)).asnumpy(), axis=1)
    for i in range(batch_size):
        if label_np[i] == ori_label[i]:
            if label_np[i] < classes - 1:
                label_np[i] += 1
            else:
                label_np[i] -= 1
    attack = JSMAAttack(net, classes, max_iteration=5)
    adv_data = attack.generate(input_np, label_np)
    assert np.any(input_np != adv_data)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_jsma_attack_2():
    """
    JSMA-Attack test
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    net = Net()
    input_shape = (1, 5)
    batch_size, classes = input_shape
    np.random.seed(5)
    input_np = np.random.random(input_shape).astype(np.float32)
    label_np = np.random.randint(classes, size=batch_size)
    ori_label = np.argmax(net(Tensor(input_np)).asnumpy(), axis=1)
    for i in range(batch_size):
        if label_np[i] == ori_label[i]:
            if label_np[i] < classes - 1:
                label_np[i] += 1
            else:
                label_np[i] -= 1
    attack = JSMAAttack(net, classes, max_iteration=5, increase=False)
    adv_data = attack.generate(input_np, label_np)
    assert np.any(input_np != adv_data)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_jsma_attack_gpu():
    """
    JSMA-Attack test
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    net = Net()
    input_shape = (1, 5)
    batch_size, classes = input_shape
    np.random.seed(5)
    input_np = np.random.random(input_shape).astype(np.float32)
    label_np = np.random.randint(classes, size=batch_size)
    ori_label = np.argmax(net(Tensor(input_np)).asnumpy(), axis=1)
    for i in range(batch_size):
        if label_np[i] == ori_label[i]:
            if label_np[i] < classes - 1:
                label_np[i] += 1
            else:
                label_np[i] -= 1
    attack = JSMAAttack(net, classes, max_iteration=5)
    adv_data = attack.generate(input_np, label_np)
    assert np.any(input_np != adv_data)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_jsma_attack_cpu():
    """
    JSMA-Attack test
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    net = Net()
    input_shape = (1, 5)
    batch_size, classes = input_shape
    np.random.seed(5)
    input_np = np.random.random(input_shape).astype(np.float32)
    label_np = np.random.randint(classes, size=batch_size)
    ori_label = np.argmax(net(Tensor(input_np)).asnumpy(), axis=1)
    for i in range(batch_size):
        if label_np[i] == ori_label[i]:
            if label_np[i] < classes - 1:
                label_np[i] += 1
            else:
                label_np[i] -= 1
    attack = JSMAAttack(net, classes, max_iteration=5)
    adv_data = attack.generate(input_np, label_np)
    assert np.any(input_np != adv_data)

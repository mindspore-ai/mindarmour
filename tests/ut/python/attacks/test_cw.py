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
CW-Attack test.
"""
import numpy as np
import pytest

import mindspore.ops.operations as M
from mindspore.nn import Cell
from mindspore import context

from mindarmour.adv_robustness.attacks import CarliniWagnerL2Attack


context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


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
def test_cw_attack():
    """
    CW-Attack test
    """
    net = Net()
    input_np = np.array([[0.1, 0.2, 0.7, 0.5, 0.4]]).astype(np.float32)
    label_np = np.array([3]).astype(np.int64)
    num_classes = input_np.shape[1]
    attack = CarliniWagnerL2Attack(net, num_classes, targeted=False)
    adv_data = attack.generate(input_np, label_np)
    assert np.any(input_np != adv_data)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_cw_attack_targeted():
    """
    CW-Attack test
    """
    net = Net()
    input_np = np.array([[0.1, 0.2, 0.7, 0.5, 0.4]]).astype(np.float32)
    target_np = np.array([1]).astype(np.int64)
    num_classes = input_np.shape[1]
    attack = CarliniWagnerL2Attack(net, num_classes, targeted=True)
    adv_data = attack.generate(input_np, target_np)
    assert np.any(input_np != adv_data)

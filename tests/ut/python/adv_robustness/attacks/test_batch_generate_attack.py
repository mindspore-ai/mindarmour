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
Batch-generate-attack test.
"""
import numpy as np
import pytest

import mindspore.ops.operations as P
from mindspore.nn import Cell
import mindspore.context as context

from mindarmour.adv_robustness.attacks import FastGradientMethod


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
        self._softmax = P.Softmax()

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
def test_batch_generate_attack():
    """
    Attack with batch-generate.
    """
    input_np = np.random.random((128, 10)).astype(np.float32)
    label = np.random.randint(0, 10, 128).astype(np.int32)
    label = np.eye(10)[label].astype(np.float32)

    attack = FastGradientMethod(Net())
    ms_adv_x = attack.batch_generate(input_np, label, batch_size=32)

    assert np.any(ms_adv_x != input_np), 'Fast gradient method: generate value' \
                                         ' must not be equal to original value.'

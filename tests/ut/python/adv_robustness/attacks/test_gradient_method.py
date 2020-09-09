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
Gradient-Attack test.
"""
import numpy as np
import pytest

import mindspore.nn as nn
from mindspore.nn import Cell
import mindspore.context as context
from mindspore.nn import SoftmaxCrossEntropyWithLogits

from mindarmour.adv_robustness.attacks import FastGradientMethod
from mindarmour.adv_robustness.attacks import FastGradientSignMethod
from mindarmour.adv_robustness.attacks import LeastLikelyClassMethod
from mindarmour.adv_robustness.attacks import RandomFastGradientMethod
from mindarmour.adv_robustness.attacks import RandomFastGradientSignMethod
from mindarmour.adv_robustness.attacks import RandomLeastLikelyClassMethod

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
def test_fast_gradient_method():
    """
    Fast gradient method unit test.
    """
    input_np = np.asarray([[0.1, 0.2, 0.7]], np.float32)
    label = np.asarray([2], np.int32)
    label = np.eye(3)[label].astype(np.float32)

    attack = FastGradientMethod(Net())
    ms_adv_x = attack.generate(input_np, label)

    assert np.any(ms_adv_x != input_np), 'Fast gradient method: generate value' \
                                         ' must not be equal to original value.'


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_inference
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_fast_gradient_method_gpu():
    """
    Fast gradient method unit test.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    input_np = np.asarray([[0.1, 0.2, 0.7]], np.float32)
    label = np.asarray([2], np.int32)
    label = np.eye(3)[label].astype(np.float32)

    attack = FastGradientMethod(Net())
    ms_adv_x = attack.generate(input_np, label)

    assert np.any(ms_adv_x != input_np), 'Fast gradient method: generate value' \
                                         ' must not be equal to original value.'


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_fast_gradient_method_cpu():
    """
    Fast gradient method unit test.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    input_np = np.asarray([[0.1, 0.2, 0.7]], np.float32)
    label = np.asarray([2], np.int32)

    loss = SoftmaxCrossEntropyWithLogits(sparse=True)
    attack = FastGradientMethod(Net(), loss_fn=loss)
    ms_adv_x = attack.generate(input_np, label)

    assert np.any(ms_adv_x != input_np), 'Fast gradient method: generate value' \
                                         ' must not be equal to original value.'


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_random_fast_gradient_method():
    """
    Random fast gradient method unit test.
    """
    input_np = np.asarray([[0.1, 0.2, 0.7]], np.float32)
    label = np.asarray([2], np.int32)
    label = np.eye(3)[label].astype(np.float32)

    attack = RandomFastGradientMethod(Net())
    ms_adv_x = attack.generate(input_np, label)

    assert np.any(ms_adv_x != input_np), 'Random fast gradient method: ' \
                                         'generate value must not be equal to' \
                                         ' original value.'


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_fast_gradient_sign_method():
    """
    Fast gradient sign method unit test.
    """
    input_np = np.asarray([[0.1, 0.2, 0.7]], np.float32)
    label = np.asarray([2], np.int32)
    label = np.eye(3)[label].astype(np.float32)

    attack = FastGradientSignMethod(Net())
    ms_adv_x = attack.generate(input_np, label)

    assert np.any(ms_adv_x != input_np), 'Fast gradient sign method: generate' \
                                         ' value must not be equal to' \
                                         ' original value.'


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_random_fast_gradient_sign_method():
    """
    Random fast gradient sign method unit test.
    """
    input_np = np.random.random((1, 28)).astype(np.float32)
    label = np.asarray([2], np.int32)
    label = np.eye(28)[label].astype(np.float32)

    attack = RandomFastGradientSignMethod(Net())
    ms_adv_x = attack.generate(input_np, label)

    assert np.any(ms_adv_x != input_np), 'Random fast gradient sign method: ' \
                                         'generate value must not be equal to' \
                                         ' original value.'


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_least_likely_class_method():
    """
    Least likely class method unit test.
    """
    input_np = np.asarray([[0.1, 0.2, 0.7]], np.float32)
    label = np.asarray([2], np.int32)
    label = np.eye(3)[label].astype(np.float32)

    attack = LeastLikelyClassMethod(Net())
    ms_adv_x = attack.generate(input_np, label)

    assert np.any(ms_adv_x != input_np), 'Least likely class method: generate' \
                                         ' value must not be equal to' \
                                         ' original value.'


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_random_least_likely_class_method():
    """
    Random least likely class method unit test.
    """
    input_np = np.asarray([[0.1, 0.2, 0.7]], np.float32)
    label = np.asarray([2], np.int32)
    label = np.eye(3)[label].astype(np.float32)

    attack = RandomLeastLikelyClassMethod(Net(), eps=0.1, alpha=0.01)
    ms_adv_x = attack.generate(input_np, label)

    assert np.any(ms_adv_x != input_np), 'Random least likely class method: ' \
                                         'generate value must not be equal to' \
                                         ' original value.'


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_assert_error():
    """
    Random least likely class method unit test.
    """
    with pytest.raises(ValueError) as e:
        assert RandomLeastLikelyClassMethod(Net(), eps=0.05, alpha=0.21)
    assert str(e.value) == 'eps must be larger than alpha!'

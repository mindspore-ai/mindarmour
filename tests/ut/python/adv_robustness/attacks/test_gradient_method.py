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
from mindspore.nn import Cell, SoftmaxCrossEntropyWithLogits
import mindspore.context as context
from mindspore.ops.composite import GradOperation

from mindarmour.adv_robustness.attacks import FastGradientMethod
from mindarmour.adv_robustness.attacks import FastGradientSignMethod
from mindarmour.adv_robustness.attacks import LeastLikelyClassMethod
from mindarmour.adv_robustness.attacks import RandomFastGradientMethod
from mindarmour.adv_robustness.attacks import RandomFastGradientSignMethod
from mindarmour.adv_robustness.attacks import RandomLeastLikelyClassMethod


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


class Net2(Cell):
    """
    Construct the network of target model. A network with multiple input data.

    Examples:
        >>> net = Net2()
    """

    def __init__(self):
        super(Net2, self).__init__()
        self._relu = nn.ReLU()

    def construct(self, inputs1, inputs2):
        out1 = self._relu(inputs1)
        out2 = self._relu(inputs2)
        return out1 + out2, out1 - out2


class LossNet(Cell):
    """
    Loss function for test.
    """
    def construct(self, loss1, loss2, labels1, labels2):
        return loss1 + loss2 - labels1 - labels2


class WithLossCell(Cell):
    """Wrap the network with loss function"""
    def __init__(self, backbone, loss_fn):
        super(WithLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn

    def construct(self, inputs1, inputs2, labels1, labels2):
        out = self._backbone(inputs1, inputs2)
        return self._loss_fn(*out, labels1, labels2)


class GradWrapWithLoss(Cell):
    """
    Construct a network to compute the gradient of loss function in \
    input space and weighted by 'weight'.
    """

    def __init__(self, network):
        super(GradWrapWithLoss, self).__init__()
        self._grad_all = GradOperation(get_all=True, sens_param=False)
        self._network = network

    def construct(self, *inputs):
        gout = self._grad_all(self._network)(*inputs)
        return gout[0]


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_fast_gradient_method():
    """
    Fast gradient method unit test.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    input_np = np.asarray([[0.1, 0.2, 0.7]], np.float32)
    label = np.asarray([2], np.int32)
    label = np.eye(3)[label].astype(np.float32)

    attack = FastGradientMethod(Net(), loss_fn=SoftmaxCrossEntropyWithLogits(sparse=False))
    ms_adv_x = attack.generate(input_np, label)

    assert np.any(ms_adv_x != input_np), 'Fast gradient method: generate value' \
                                         ' must not be equal to original value.'


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
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

    attack = FastGradientMethod(Net(), loss_fn=SoftmaxCrossEntropyWithLogits(sparse=False))
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
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    input_np = np.asarray([[0.1, 0.2, 0.7]], np.float32)
    label = np.asarray([2], np.int32)
    label = np.eye(3)[label].astype(np.float32)

    attack = RandomFastGradientMethod(Net(), loss_fn=SoftmaxCrossEntropyWithLogits(sparse=False))
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
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    input_np = np.asarray([[0.1, 0.2, 0.7]], np.float32)
    label = np.asarray([2], np.int32)
    label = np.eye(3)[label].astype(np.float32)

    attack = FastGradientSignMethod(Net(), loss_fn=SoftmaxCrossEntropyWithLogits(sparse=False))
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
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    input_np = np.random.random((1, 28)).astype(np.float32)
    label = np.asarray([2], np.int32)
    label = np.eye(28)[label].astype(np.float32)

    attack = RandomFastGradientSignMethod(Net(), loss_fn=SoftmaxCrossEntropyWithLogits(sparse=False))
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
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    input_np = np.asarray([[0.1, 0.2, 0.7]], np.float32)
    label = np.asarray([2], np.int32)
    label = np.eye(3)[label].astype(np.float32)

    attack = LeastLikelyClassMethod(Net(), loss_fn=SoftmaxCrossEntropyWithLogits(sparse=False))
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
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    input_np = np.asarray([[0.1, 0.2, 0.7]], np.float32)
    label = np.asarray([2], np.int32)
    label = np.eye(3)[label].astype(np.float32)

    attack = RandomLeastLikelyClassMethod(Net(), eps=0.1, alpha=0.01, \
        loss_fn=SoftmaxCrossEntropyWithLogits(sparse=False))
    ms_adv_x = attack.generate(input_np, label)

    assert np.any(ms_adv_x != input_np), 'Random least likely class method: ' \
                                         'generate value must not be equal to' \
                                         ' original value.'


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_fast_gradient_method_multi_inputs():
    """
    Fast gradient method unit test.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    inputs1 = np.asarray([[0.1, 0.2, 0.7]]).astype(np.float32)
    inputs2 = np.asarray([[0.4, 0.8, 0.5]]).astype(np.float32)
    labels1 = np.expand_dims(np.eye(3)[1].astype(np.float32), axis=0)
    labels2 = np.expand_dims(np.eye(3)[2].astype(np.float32), axis=0)

    with_loss_cell = WithLossCell(Net2(), LossNet())
    grad_with_loss_net = GradWrapWithLoss(with_loss_cell)
    attack = FastGradientMethod(grad_with_loss_net)
    ms_adv_x = attack.generate((inputs1, inputs2), (labels1, labels2))

    assert np.any(ms_adv_x != inputs1), 'Fast gradient method: generate value' \
                                         ' must not be equal to original value.'


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_assert_error():
    """
    Random least likely class method unit test.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    with pytest.raises(ValueError) as e:
        assert RandomLeastLikelyClassMethod(Net(), eps=0.05, alpha=0.21, \
            loss_fn=SoftmaxCrossEntropyWithLogits(sparse=False))
    assert str(e.value) == 'eps must be larger than alpha!'

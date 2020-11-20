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

import mindspore.context as context
import mindspore.ops.operations as P
from mindspore.ops.composite import GradOperation
from mindspore.nn import Cell, SoftmaxCrossEntropyWithLogits

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


class Net2(Cell):
    """
    Construct the network of target model. A network with multiple input data.

    Examples:
        >>> net = Net2()
    """

    def __init__(self):
        super(Net2, self).__init__()
        self._softmax = P.Softmax()

    def construct(self, inputs1, inputs2):
        out1 = self._softmax(inputs1)
        out2 = self._softmax(inputs2)
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
def test_batch_generate_attack():
    """
    Attack with batch-generate.
    """
    input_np = np.random.random((128, 10)).astype(np.float32)
    label = np.random.randint(0, 10, 128).astype(np.int32)
    label = np.eye(10)[label].astype(np.float32)

    attack = FastGradientMethod(Net(), loss_fn=SoftmaxCrossEntropyWithLogits(sparse=False))
    ms_adv_x = attack.batch_generate(input_np, label, batch_size=32)

    assert np.any(ms_adv_x != input_np), 'Fast gradient method: generate value' \
                                            ' must not be equal to original value.'


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_batch_generate_attack_multi_inputs():
    """
    Attack with batch-generate by multi-inputs.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    inputs1 = np.random.random((128, 10)).astype(np.float32)
    inputs2 = np.random.random((128, 10)).astype(np.float32)
    labels1 = np.random.randint(0, 10, 128).astype(np.int32)
    labels2 = np.random.randint(0, 10, 128).astype(np.int32)
    labels1 = np.eye(10)[labels1].astype(np.float32)
    labels2 = np.eye(10)[labels2].astype(np.float32)

    with_loss_cell = WithLossCell(Net2(), LossNet())
    grad_with_loss_net = GradWrapWithLoss(with_loss_cell)
    attack = FastGradientMethod(grad_with_loss_net)
    ms_adv_x = attack.batch_generate((inputs1, inputs2), (labels1, labels2), batch_size=32)

    assert np.any(ms_adv_x != inputs1), 'Fast gradient method: generate value' \
                                         ' must not be equal to original value.'

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
DeepFool-Attack test.
"""
import numpy as np
import pytest

import mindspore.ops.operations as P
from mindspore.nn import Cell
from mindspore import context
from mindspore import Tensor

from mindarmour.adv_robustness.attacks import DeepFool

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
    Construct the network of target model, specifically for detection model test case.

    Examples:
        >>> net = Net2()
    """
    def __init__(self):
        super(Net2, self).__init__()
        self._softmax = P.Softmax()

    def construct(self, inputs1, inputs2):
        out1 = self._softmax(inputs1)
        out2 = self._softmax(inputs2)
        return out2, out1


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_deepfool_attack():
    """
    Deepfool-Attack test
    """
    net = Net()
    input_shape = (1, 5)
    _, classes = input_shape
    input_np = np.array([[0.1, 0.2, 0.7, 0.5, 0.4]]).astype(np.float32)
    input_me = Tensor(input_np)
    true_labels = np.argmax(net(input_me).asnumpy(), axis=1)
    attack = DeepFool(net, classes, max_iters=10, norm_level=2,
                      bounds=(0.0, 1.0))
    adv_data = attack.generate(input_np, true_labels)
    # expected adv value
    expect_value = np.asarray([[0.10300991, 0.20332647, 0.59308802, 0.59651263,
                                0.40406296]])
    assert np.allclose(adv_data, expect_value), 'mindspore deepfool_method' \
        ' implementation error, ms_adv_x != expect_value'


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_deepfool_attack_detection():
    """
    Deepfool-Attack test
    """
    net = Net2()
    inputs1_np = np.random.random((2, 10, 10)).astype(np.float32)
    inputs2_np = np.random.random((2, 10, 5)).astype(np.float32)
    gt_boxes, gt_logits = net(Tensor(inputs1_np), Tensor(inputs2_np))
    gt_boxes, gt_logits = gt_boxes.asnumpy(), gt_logits.asnumpy()
    gt_labels = np.argmax(gt_logits, axis=2)
    num_classes = 10

    attack = DeepFool(net, num_classes, model_type='detection', reserve_ratio=0.3,
                      bounds=(0.0, 1.0))
    adv_data = attack.generate((inputs1_np, inputs2_np), (gt_boxes, gt_labels))
    assert np.any(adv_data != inputs1_np)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_deepfool_attack_inf():
    """
    Deepfool-Attack test
    """
    net = Net()
    input_shape = (1, 5)
    _, classes = input_shape
    input_np = np.array([[0.1, 0.2, 0.7, 0.5, 0.4]]).astype(np.float32)
    input_me = Tensor(input_np)
    true_labels = np.argmax(net(input_me).asnumpy(), axis=1)
    attack = DeepFool(net, classes, max_iters=10, norm_level=np.inf,
                      bounds=(0.0, 1.0))
    adv_data = attack.generate(input_np, true_labels)
    assert np.any(input_np != adv_data)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_value_error():
    net = Net()
    input_shape = (1, 5)
    _, classes = input_shape
    input_np = np.array([[0.1, 0.2, 0.7, 0.5, 0.4]]).astype(np.float32)
    input_me = Tensor(input_np)
    true_labels = np.argmax(net(input_me).asnumpy(), axis=1)
    with pytest.raises(NotImplementedError):
        # norm_level=0 is not available
        attack = DeepFool(net, classes, max_iters=10, norm_level=1,
                          bounds=(0.0, 1.0))
        assert attack.generate(input_np, true_labels)

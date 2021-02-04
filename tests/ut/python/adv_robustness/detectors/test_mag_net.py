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
Mag-net detector test.
"""
import numpy as np
import pytest

import mindspore.ops.operations as P
from mindspore.nn import Cell
from mindspore.ops.operations import Add
from mindspore import Model
from mindspore import context

from mindarmour.adv_robustness.detectors import ErrorBasedDetector
from mindarmour.adv_robustness.detectors import DivergenceBasedDetector

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


class Net(Cell):
    """
    Construct the network of target model.
    """

    def __init__(self):
        super(Net, self).__init__()
        self.add = Add()

    def construct(self, inputs):
        """
        Construct network.

        Args:
            inputs (Tensor): Input data.
        """
        return self.add(inputs, inputs)


class PredNet(Cell):
    """
    Construct the network of target model.
    """

    def __init__(self):
        super(PredNet, self).__init__()
        self.shape = P.Shape()
        self.reshape = P.Reshape()
        self._softmax = P.Softmax()

    def construct(self, inputs):
        """
        Construct network.

        Args:
            inputs (Tensor): Input data.
        """
        data = self.reshape(inputs, (self.shape(inputs)[0], -1))
        return self._softmax(data)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_mag_net():
    """
    Compute mindspore result.
    """
    np.random.seed(5)
    ori = np.random.rand(4, 4, 4).astype(np.float32)
    np.random.seed(6)
    adv = np.random.rand(4, 4, 4).astype(np.float32)
    model = Model(Net())
    detector = ErrorBasedDetector(model)
    detector.fit(ori)
    detected_res = detector.detect(adv)
    expected_value = np.array([1, 1, 1, 1])
    assert np.all(detected_res == expected_value)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_mag_net_transform():
    """
    Compute mindspore result.
    """
    np.random.seed(6)
    adv = np.random.rand(4, 4, 4).astype(np.float32)
    model = Model(Net())
    detector = ErrorBasedDetector(model)
    adv_trans = detector.transform(adv)
    assert np.any(adv_trans != adv)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_mag_net_divergence():
    """
    Compute mindspore result.
    """
    np.random.seed(5)
    ori = np.random.rand(4, 4, 4).astype(np.float32)
    np.random.seed(6)
    adv = np.random.rand(4, 4, 4).astype(np.float32)
    encoder = Model(Net())
    model = Model(PredNet())
    detector = DivergenceBasedDetector(encoder, model)
    threshold = detector.fit(ori)
    detector.set_threshold(threshold)
    detected_res = detector.detect(adv)
    expected_value = np.array([1, 0, 1, 1])
    assert np.all(detected_res == expected_value)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_mag_net_divergence_transform():
    """
    Compute mindspore result.
    """
    np.random.seed(6)
    adv = np.random.rand(4, 4, 4).astype(np.float32)
    encoder = Model(Net())
    model = Model(PredNet())
    detector = DivergenceBasedDetector(encoder, model)
    adv_trans = detector.transform(adv)
    assert np.any(adv_trans != adv)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_value_error():
    np.random.seed(6)
    adv = np.random.rand(4, 4, 4).astype(np.float32)
    encoder = Model(Net())
    model = Model(PredNet())
    detector = DivergenceBasedDetector(encoder, model, option='bad_op')
    with pytest.raises(NotImplementedError):
        assert detector.detect_diff(adv)

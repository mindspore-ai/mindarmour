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
EnsembleDetector Test
"""
import numpy as np
import pytest

from mindspore.nn import Cell
from mindspore.ops.operations import Add
from mindspore.train.model import Model
from mindspore import context

from mindarmour.adv_robustness.detectors import ErrorBasedDetector
from mindarmour.adv_robustness.detectors import RegionBasedDetector
from mindarmour.adv_robustness.detectors import EnsembleDetector

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


class AutoNet(Cell):
    """
    Construct the network of target model.
    """
    def __init__(self):
        super(AutoNet, self).__init__()
        self.add = Add()

    def construct(self, inputs):
        """
        Construct network.

        Args:
            inputs (Tensor): Input data.
        """
        return self.add(inputs, inputs)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_ensemble_detector():
    """
    Compute mindspore result.
    """
    np.random.seed(6)
    adv = np.random.rand(4, 4).astype(np.float32)
    model = Model(Net())
    auto_encoder = Model(AutoNet())
    random_label = np.random.randint(10, size=4)
    labels = np.eye(10)[random_label]
    magnet_detector = ErrorBasedDetector(auto_encoder)
    region_detector = RegionBasedDetector(model)
    # use this to enable radius in region_detector
    region_detector.fit(adv, labels)
    detectors = [magnet_detector, region_detector]
    detector = EnsembleDetector(detectors)
    detected_res = detector.detect(adv)
    expected_value = np.array([0, 1, 0, 0])
    assert np.all(detected_res == expected_value)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_error():
    np.random.seed(6)
    adv = np.random.rand(4, 4).astype(np.float32)
    model = Model(Net())
    auto_encoder = Model(AutoNet())
    random_label = np.random.randint(10, size=4)
    labels = np.eye(10)[random_label]
    magnet_detector = ErrorBasedDetector(auto_encoder)
    region_detector = RegionBasedDetector(model)
    # use this to enable radius in region_detector
    detectors = [magnet_detector, region_detector]
    detector = EnsembleDetector(detectors)
    with pytest.raises(NotImplementedError):
        assert detector.fit(adv, labels)

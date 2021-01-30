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
Region-based detector test.
"""
import numpy as np
import pytest

from mindspore.nn import Cell
from mindspore import Model
from mindspore import context
from mindspore.ops.operations import Add

from mindarmour.adv_robustness.detectors import RegionBasedDetector


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


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_region_based_classification():
    """
    Compute mindspore result.
    """
    np.random.seed(5)
    ori = np.random.rand(4, 4).astype(np.float32)
    labels = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0],
                       [0, 1, 0, 0]]).astype(np.int32)
    np.random.seed(6)
    adv = np.random.rand(4, 4).astype(np.float32)
    model = Model(Net())
    detector = RegionBasedDetector(model)
    radius = detector.fit(ori, labels)
    detector.set_radius(radius)
    detected_res = detector.detect(adv)
    expected_value = np.array([0, 0, 1, 0])
    assert np.all(detected_res == expected_value)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_value_error():
    np.random.seed(5)
    ori = np.random.rand(4, 4).astype(np.float32)
    labels = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0],
                       [0, 1, 0, 0]]).astype(np.int32)
    np.random.seed(6)
    adv = np.random.rand(4, 4).astype(np.float32)
    model = Model(Net())
    # model should be mindspore model
    with pytest.raises(TypeError):
        assert RegionBasedDetector(Net())

    with pytest.raises(ValueError):
        assert RegionBasedDetector(model, number_points=-1)

    with pytest.raises(ValueError):
        assert RegionBasedDetector(model, initial_radius=-1)

    with pytest.raises(ValueError):
        assert RegionBasedDetector(model, max_radius=-2.2)

    with pytest.raises(ValueError):
        assert RegionBasedDetector(model, search_step=0)

    with pytest.raises(TypeError):
        assert RegionBasedDetector(model, sparse='False')

    detector = RegionBasedDetector(model)
    with pytest.raises(TypeError):
        # radius must not empty
        assert detector.detect(adv)

    radius = detector.fit(ori, labels)
    detector.set_radius(radius)
    with pytest.raises(TypeError):
        # adv type should be in (list, tuple, numpy.ndarray)
        assert detector.detect(adv.tostring())

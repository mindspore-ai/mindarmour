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
Spatial-smoothing detector test.
"""
import numpy as np
import pytest

import mindspore.ops.operations as P
from mindspore import Model
from mindspore.nn import Cell
from mindspore import context

from mindarmour.adv_robustness.detectors import SpatialSmoothing

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


# for use
class Net(Cell):
    """
    Construct the network of target model.
    """
    def __init__(self):
        super(Net, self).__init__()
        self._softmax = P.Softmax()

    def construct(self, inputs):
        """
        Construct network.

        Args:
            inputs (Tensor): Input data.
        """
        return self._softmax(inputs)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_spatial_smoothing():
    """
    Compute mindspore result.
    """
    input_shape = (50, 3)

    np.random.seed(1)
    input_np = np.random.randn(*input_shape).astype(np.float32)

    np.random.seed(2)
    adv_np = np.random.randn(*input_shape).astype(np.float32)

    # mock user model
    model = Model(Net())
    detector = SpatialSmoothing(model)
    # Training
    threshold = detector.fit(input_np)
    detector.set_threshold(threshold.item())
    detected_res = np.array(detector.detect(adv_np))
    idx = np.where(detected_res > 0)
    expected_value = np.array([10, 39, 48])
    assert np.all(idx == expected_value)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_spatial_smoothing_diff():
    """
        Compute mindspore result.
    """
    input_shape = (50, 3)
    np.random.seed(1)
    input_np = np.random.randn(*input_shape).astype(np.float32)

    np.random.seed(2)
    adv_np = np.random.randn(*input_shape).astype(np.float32)

    # mock user model
    model = Model(Net())
    detector = SpatialSmoothing(model)
    # Training
    detector.fit(input_np)
    diffs = detector.detect_diff(adv_np)
    expected_value = np.array([0.20959496, 0.69537306, 0.13034256, 0.7421039,
                               0.41419053, 0.56346416, 0.4277994, 0.3240941,
                               0.048190027, 0.6806958, 1.1405756, 0.587804,
                               0.40533313, 0.2875523, 0.36801508, 0.61993587,
                               0.49286827, 0.13222921, 0.68012404, 0.4164942,
                               0.25758877, 0.6008735, 0.60623455, 0.34981924,
                               0.3945489, 0.879787, 0.3934811, 0.23387678,
                               0.63480926, 0.56435543, 0.16067612, 0.57489645,
                               0.21772699, 0.55924356, 0.5186635, 0.7094835,
                               0.0613693, 0.13305652, 0.11505881, 1.2404268,
                               0.50948, 0.15797901, 0.44473758, 0.2495422,
                               0.38254014, 0.543059, 0.06452079, 0.36902517,
                               1.1845329, 0.3870097])
    assert np.allclose(diffs, expected_value, 0.0001, 0.0001)

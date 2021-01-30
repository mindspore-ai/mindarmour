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
Similarity-detector test.
"""
import numpy as np
import pytest

from mindspore.nn import Cell
from mindspore import Model
from mindspore import context
from mindspore.ops.operations import Add

from mindarmour.adv_robustness.detectors import SimilarityDetector

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


class EncoderNet(Cell):
    """
    Similarity encoder for input data
    """

    def __init__(self, encode_dim):
        super(EncoderNet, self).__init__()
        self._encode_dim = encode_dim
        self.add = Add()

    def construct(self, inputs):
        """
        construct the neural network
        Args:
            inputs (Tensor): input data to neural network.
        Returns:
            Tensor, output of neural network.
        """
        return self.add(inputs, inputs)

    def get_encode_dim(self):
        """
        Get the dimension of encoded inputs

        Returns:
            int, dimension of encoded inputs.
        """
        return self._encode_dim


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_similarity_detector():
    """
    Similarity detector unit test
    """
    # Prepare dataset
    np.random.seed(5)
    x_train = np.random.rand(1000, 32, 32, 3).astype(np.float32)
    perm = np.random.permutation(x_train.shape[0])

    # Generate query sequences
    benign_queries = x_train[perm[:1000], :, :, :]
    suspicious_queries = x_train[perm[-1], :, :, :] + np.random.normal(
        0, 0.05, (1000,) + x_train.shape[1:])
    suspicious_queries = suspicious_queries.astype(np.float32)

    # explicit threshold not provided, calculate threshold for K
    encoder = Model(EncoderNet(encode_dim=256))
    detector = SimilarityDetector(max_k_neighbor=50, trans_model=encoder)
    num_nearest_neighbors, thresholds = detector.fit(inputs=x_train)
    detector.set_threshold(num_nearest_neighbors[-1], thresholds[-1])

    detector.detect(benign_queries)
    detections = detector.get_detection_interval()
    # compare
    expected_value = 0
    assert len(detections) == expected_value

    detector.clear_buffer()
    detector.detect(suspicious_queries)

    # compare
    expected_value = [1051, 1102, 1153, 1204, 1255,
                      1306, 1357, 1408, 1459, 1510,
                      1561, 1612, 1663, 1714, 1765,
                      1816, 1867, 1918, 1969]
    assert np.all(detector.get_detected_queries() == expected_value)

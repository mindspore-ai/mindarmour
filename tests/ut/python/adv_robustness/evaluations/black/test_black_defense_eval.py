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
Black-box defense evaluation test.
"""
import numpy as np
import pytest

from mindarmour.adv_robustness.evaluations import BlackDefenseEvaluate


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_def_eval():
    """
    Tests for black-box defense evaluation
    """
    # prepare data
    raw_preds = np.array([[0.1, 0.1, 0.2, 0.6], [0.1, 0.7, 0.0, 0.2],
                          [0.8, 0.1, 0.0, 0.1], [0.1, 0.1, 0.2, 0.6],
                          [0.1, 0.7, 0.0, 0.2], [0.8, 0.1, 0.0, 0.1],
                          [0.1, 0.1, 0.2, 0.6], [0.1, 0.7, 0.0, 0.2],
                          [0.8, 0.1, 0.0, 0.1], [0.1, 0.1, 0.2, 0.6]])

    def_preds = np.array([[0.1, 0.1, 0.2, 0.6], [0.1, 0.7, 0.0, 0.2],
                          [0.8, 0.1, 0.0, 0.1], [0.1, 0.1, 0.2, 0.6],
                          [0.1, 0.7, 0.0, 0.2], [0.8, 0.1, 0.0, 0.1],
                          [0.1, 0.1, 0.2, 0.6], [0.1, 0.7, 0.0, 0.2],
                          [0.8, 0.1, 0.0, 0.1], [0.1, 0.1, 0.2, 0.6]])
    raw_query_counts = np.array([0, 0, 0, 0, 0, 10, 10, 20, 20, 30])
    def_query_counts = np.array([0, 0, 0, 0, 0, 30, 30, 40, 40, 50])

    raw_query_time = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 2, 2, 4, 4, 6])
    def_query_time = np.array([0.3, 0.3, 0.3, 0.3, 0.3, 4, 4, 8, 8, 12])

    def_detection_counts = np.array([1, 0, 0, 0, 1, 5, 5, 5, 10, 20])

    true_labels = np.array([3, 1, 0, 3, 1, 0, 3, 1, 0, 3])

    # create obj
    def_eval = BlackDefenseEvaluate(raw_preds,
                                    def_preds,
                                    raw_query_counts,
                                    def_query_counts,
                                    raw_query_time,
                                    def_query_time,
                                    def_detection_counts,
                                    true_labels,
                                    max_queries=100)
    # run eval
    qcv = def_eval.qcv()
    asv = def_eval.asv()
    fpr = def_eval.fpr()
    qrv = def_eval.qrv()
    res = [qcv, asv, fpr, qrv]

    # compare
    expected_value = [0.2, 0.0, 0.4, 2.0]
    assert np.allclose(res, expected_value, 0.0001, 0.0001)

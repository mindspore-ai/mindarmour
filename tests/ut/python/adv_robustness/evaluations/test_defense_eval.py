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
Defense evaluation test.
"""
import numpy as np
import pytest

from mindarmour.adv_robustness.evaluations import DefenseEvaluate


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_def_eval():
    # prepare data
    raw_preds = np.array([[0.1, 0.1, 0.2, 0.6],
                          [0.1, 0.7, 0.0, 0.2],
                          [0.8, 0.1, 0.0, 0.1]])
    def_preds = np.array([[0.1, 0.1, 0.1, 0.7],
                          [0.1, 0.6, 0.2, 0.1],
                          [0.1, 0.2, 0.1, 0.6]])
    true_labels = np.array([3, 1, 0])

    # create obj
    def_eval = DefenseEvaluate(raw_preds, def_preds, true_labels)

    # run eval
    cav = def_eval.cav()
    crr = def_eval.crr()
    csr = def_eval.csr()
    ccv = def_eval.ccv()
    cos = def_eval.cos()
    res = [cav, crr, csr, ccv, cos]

    # compare
    expected_value = [-0.3333, 0.0, 0.3333, 0.0999, 0.0450]
    assert np.allclose(res, expected_value, 0.0001, 0.0001)

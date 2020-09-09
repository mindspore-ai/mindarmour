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
Attack evaluation test.
"""
import numpy as np
import pytest

from mindarmour.adv_robustness.evaluations import AttackEvaluate


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_attack_eval():
    # prepare test data
    np.random.seed(1024)
    inputs = np.random.normal(size=(3, 512, 512, 3))
    labels = np.array([[0.1, 0.1, 0.2, 0.6],
                       [0.1, 0.7, 0.0, 0.2],
                       [0.8, 0.1, 0.0, 0.1]])
    adv_x = inputs + np.ones((3, 512, 512, 3))*0.001
    adv_y = np.array([[0.1, 0.1, 0.2, 0.6],
                      [0.1, 0.0, 0.8, 0.1],
                      [0.0, 0.9, 0.1, 0.0]])

    # create obj
    attack_eval = AttackEvaluate(inputs, labels, adv_x, adv_y)

    # run eval
    mr = attack_eval.mis_classification_rate()
    acac = attack_eval.avg_conf_adv_class()
    l_0, l_2, l_inf = attack_eval.avg_lp_distance()
    ass = attack_eval.avg_ssim()
    nte = attack_eval.nte()
    res = [mr, acac, l_0, l_2, l_inf, ass, nte]

    # compare
    expected_value = [0.6666, 0.8500, 1.0, 0.0009, 0.0001, 0.9999, 0.75]
    assert np.allclose(res, expected_value, 0.0001, 0.0001)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_value_error():
    # prepare test data
    np.random.seed(1024)
    inputs = np.random.normal(size=(3, 512, 512, 3))
    labels = np.array([[0.1, 0.1, 0.2, 0.6],
                       [0.1, 0.7, 0.0, 0.2],
                       [0.8, 0.1, 0.0, 0.1]])
    adv_x = inputs + np.ones((3, 512, 512, 3))*0.001
    adv_y = np.array([[0.1, 0.1, 0.2, 0.6],
                      [0.1, 0.0, 0.8, 0.1],
                      [0.0, 0.9, 0.1, 0.0]])

    # create obj
    with pytest.raises(ValueError) as e:
        assert AttackEvaluate(inputs, labels, adv_x, adv_y, targeted=True)
    assert str(e.value) == 'targeted attack need target_label, but got None.'


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_empty_input_error():
    # prepare test data
    np.random.seed(1024)
    inputs = np.array([])
    labels = np.array([])
    adv_x = inputs
    adv_y = np.array([])

    # create obj
    with pytest.raises(ValueError) as e:
        assert AttackEvaluate(inputs, labels, adv_x, adv_y)
    assert str(e.value) == 'inputs must not be empty'

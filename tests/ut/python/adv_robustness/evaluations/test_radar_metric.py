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
Radar map test.
"""
import numpy as np
import pytest
from mindarmour.adv_robustness.evaluations import RadarMetric


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_radar_metric():
    # prepare data
    metrics_name = ['MR', 'ACAC', 'ASS', 'NTE', 'RGB']
    def_metrics = [0.9, 0.85, 0.6, 0.7, 0.8]
    raw_metrics = [0.5, 0.3, 0.55, 0.65, 0.7]
    metrics_data = np.array([def_metrics, raw_metrics])
    metrics_labels = ['before', 'after']

    # create obj
    _ = RadarMetric(metrics_name, metrics_data, metrics_labels, title='',
                    scale='sparse')


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_value_error():
    # prepare data
    metrics_name = ['MR', 'ACAC', 'ASS', 'NTE', 'RGB']
    def_metrics = [0.9, 0.85, 0.6, 0.7, 0.8]
    raw_metrics = [0.5, 0.3, 0.55, 0.65, 0.7]
    metrics_data = np.array([def_metrics, raw_metrics])
    metrics_labels = ['before', 'after']

    with pytest.raises(ValueError):
        assert RadarMetric(metrics_name, metrics_data, metrics_labels,
                           title='', scale='bad_s')

    with pytest.raises(ValueError):
        assert RadarMetric(['MR', 'ACAC', 'ASS'], metrics_data, metrics_labels,
                           title='', scale='bad_s')

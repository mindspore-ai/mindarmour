# Copyright 2021 Huawei Technologies Co., Ltd
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
Concept drift test.
"""


import logging
import pytest
import numpy as np
from mindarmour import ConceptDriftCheckTimeSeries
from mindarmour.utils.logger import LogUtil

LOGGER = LogUtil.get_instance()
TAG = 'Concept_Test'


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_cp():
    """
    Concept drift test.
    """
    # create data
    data = 5*np.random.rand(1000)
    data[200: 800] = 50*np.random.rand(600)
    # initialization
    concept = ConceptDriftCheckTimeSeries(window_size=100, rolling_window=10,
                                          step=10, threshold_index=1.5, need_label=False)
    # drift check
    drift_score, threshold, concept_drift_location = concept.concept_check(data)
    LOGGER.set_level(logging.DEBUG)
    LOGGER.debug(TAG, '--start concept drift test--')
    LOGGER.debug(threshold, '--concept drift threshold--')
    LOGGER.debug(concept_drift_location, '--concept drift location--')
    LOGGER.debug(TAG, '--end concept drift test--')
    assert np.any(drift_score >= 0.0)

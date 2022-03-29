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
Concept drift test for images.
"""

import os
import logging
import pytest
import numpy as np
from mindarmour.utils.logger import LogUtil
from mindarmour.reliability.concept_drift.concept_drift_check_images import OodDetectorFeatureCluster
from mindspore import Model, context
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from examples.common.networks.lenet5.lenet5_net_for_fuzzing import LeNet5

LOGGER = LogUtil.get_instance()
TAG = 'Concept_Test'



@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_concept_drift_image_ascend():
    """
    Feature: test concept drift with images
    Description: make sure the odd detector working properly
    Expectation: assert np.any(result >=0.0)
    """
    # load model
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    cur_path = os.path.abspath(os.path.dirname(__file__))
    ckpt_path = '../../dataset/trained_ckpt_file/checkpoint_lenet-10_1875.ckpt'
    ckpt_path = os.path.join(cur_path, ckpt_path)
    net = LeNet5()
    load_dict = load_checkpoint(ckpt_path)
    load_param_into_net(net, load_dict)
    model = Model(net)
    # load data
    ds_train = np.load(os.path.join(cur_path, '../../dataset/concept_train_lenet.npy'))
    ds_eval = np.load(os.path.join(cur_path, '../../dataset/concept_test_lenet1.npy'))
    ds_test = np.load(os.path.join(cur_path, '../../dataset/concept_test_lenet2.npy'))
    # ood detector initialization
    detector = OodDetectorFeatureCluster(model, ds_train, n_cluster=10, layer='output[:Tensor]')
    # get optimal threshold with ds_eval
    num = int(len(ds_eval) / 2)
    label = np.concatenate((np.zeros(num), np.ones(num)), axis=0)  # ID data = 0, OOD data = 1
    optimal_threshold = detector.get_optimal_threshold(label, ds_eval)
    # get result of ds_test. We can also set threshold by ourselves.
    result = detector.ood_predict(optimal_threshold, ds_test)
    # result log
    LOGGER.set_level(logging.DEBUG)
    LOGGER.debug(TAG, '--start ood test--')
    LOGGER.debug(result, '--ood result--')
    LOGGER.debug(optimal_threshold, '--the optimal threshold--')
    LOGGER.debug(TAG, '--end ood test--')
    assert np.any(result >= 0.0)

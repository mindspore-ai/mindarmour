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
LBFGS-Attack test.
"""
import os

import numpy as np
import pytest
from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from mindarmour.adv_robustness.attacks import LBFGS
from mindarmour.utils.logger import LogUtil

from tests.ut.python.utils.mock_net import Net

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


LOGGER = LogUtil.get_instance()
TAG = 'LBFGS_Test'
LOGGER.set_level('DEBUG')


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_lbfgs_attack():
    """
    LBFGS-Attack test
    """
    np.random.seed(123)
    # upload trained network
    current_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_path = os.path.join(current_dir,
                             '../../dataset/trained_ckpt_file/checkpoint_lenet-10_1875.ckpt')
    net = Net()
    load_dict = load_checkpoint(ckpt_path)
    load_param_into_net(net, load_dict)

    # get one mnist image
    input_np = np.load(os.path.join(current_dir,
                                    '../../dataset/test_images.npy'))[:1]
    label_np = np.load(os.path.join(current_dir,
                                    '../../dataset/test_labels.npy'))[:1]
    LOGGER.debug(TAG, 'true label is :{}'.format(label_np[0]))
    classes = 10
    target_np = np.random.randint(0, classes, 1)
    while target_np == label_np[0]:
        target_np = np.random.randint(0, classes)
    target_np = np.eye(10)[target_np].astype(np.float32)

    attack = LBFGS(net, is_targeted=True)
    LOGGER.debug(TAG, 'target_np is :{}'.format(target_np[0]))
    _ = attack.generate(input_np, target_np)

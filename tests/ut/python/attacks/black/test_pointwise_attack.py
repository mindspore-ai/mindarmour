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
PointWise Attack test
"""
import os
import sys

import numpy as np
import pytest
from mindspore import Tensor
from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from mindarmour import BlackModel
from mindarmour.adv_robustness.attacks import PointWiseAttack
from mindarmour.utils.logger import LogUtil

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "../../../../../"))
from example.mnist_demo.lenet5_net import LeNet5


context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

LOGGER = LogUtil.get_instance()
TAG = 'Pointwise_Test'
LOGGER.set_level('INFO')


class ModelToBeAttacked(BlackModel):
    """model to be attack"""

    def __init__(self, network):
        super(ModelToBeAttacked, self).__init__()
        self._network = network

    def predict(self, inputs):
        """predict"""
        result = self._network(Tensor(inputs.astype(np.float32)))
        return result.asnumpy()


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_pointwise_attack_method():
    """
    Pointwise attack method unit test.
    """
    np.random.seed(123)
    # upload trained network
    current_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_name = os.path.join(current_dir,
                             '../../test_data/trained_ckpt_file/checkpoint_lenet-10_1875.ckpt')
    net = LeNet5()
    load_dict = load_checkpoint(ckpt_name)
    load_param_into_net(net, load_dict)

    # get one mnist image
    input_np = np.load(os.path.join(current_dir,
                                    '../../test_data/test_images.npy'))[:3]
    labels = np.load(os.path.join(current_dir,
                                  '../../test_data/test_labels.npy'))[:3]
    model = ModelToBeAttacked(net)
    pre_label = np.argmax(model.predict(input_np), axis=1)
    LOGGER.info(TAG, 'original sample predict labels are :{}'.format(pre_label))
    LOGGER.info(TAG, 'true labels are: {}'.format(labels))
    attack = PointWiseAttack(model, sparse=True, is_targeted=False)
    is_adv, adv_data, _ = attack.generate(input_np, pre_label)
    LOGGER.info(TAG, 'adv sample predict labels are: {}'
                .format(np.argmax(model.predict(adv_data), axis=1)))

    assert np.any(adv_data[is_adv][0] != input_np[is_adv][0]), 'Pointwise attack method: ' \
                                             'generate value must not be equal' \
                                             ' to original value.'

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
SaltAndPepper Attack Test
"""
import os
import numpy as np
import pytest

from mindspore import Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore import context

from mindarmour import BlackModel
from mindarmour.adv_robustness.attacks import SaltAndPepperNoiseAttack
from tests.ut.python.utils.mock_net import Net

context.set_context(mode=context.GRAPH_MODE)
context.set_context(device_target="Ascend")


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
def test_salt_and_pepper_attack_method():
    """
    Salt and pepper attack method unit test.
    """
    np.random.seed(123)
    # upload trained network
    current_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_path = os.path.join(current_dir,
                             '../../../dataset/trained_ckpt_file/checkpoint_lenet-10_1875.ckpt')
    net = Net()
    load_dict = load_checkpoint(ckpt_path)
    load_param_into_net(net, load_dict)

    # get one mnist image
    inputs = np.load(os.path.join(current_dir, '../../../dataset/test_images.npy'))[:3]
    labels = np.load(os.path.join(current_dir, '../../../dataset/test_labels.npy'))[:3]
    model = ModelToBeAttacked(net)

    attack = SaltAndPepperNoiseAttack(model, sparse=True)
    _, adv_data, _ = attack.generate(inputs, labels)
    assert np.any(adv_data[0] != inputs[0]), 'Salt and pepper attack method:  generate value must not be equal' \
                                             ' to original value.'

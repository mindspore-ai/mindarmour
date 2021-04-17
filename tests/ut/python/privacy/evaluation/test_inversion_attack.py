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
Inversion attack test
"""
import pytest

import numpy as np

import mindspore.context as context

from mindarmour.privacy.evaluation.inversion_attack import ImageInversionAttack

from tests.ut.python.utils.mock_net import Net


@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
@pytest.mark.component_mindarmour
def test_inversion_attack_graph():
    context.set_context(mode=context.GRAPH_MODE)
    net = Net()
    original_images = np.random.random((2, 1, 32, 32)).astype(np.float32)
    target_features = np.random.random((2, 10)).astype(np.float32)
    inversion_attack = ImageInversionAttack(net, input_shape=(1, 32, 32), input_bound=(0, 1), loss_weights=[1, 0.2, 5])
    inversion_images = inversion_attack.generate(target_features, iters=10)
    avg_ssim = inversion_attack.evaluate(original_images, inversion_images)
    assert 0 < avg_ssim[1] < 1
    assert target_features.shape[0] == inversion_images.shape[0]


@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
@pytest.mark.component_mindarmour
def test_inversion_attack_pynative():
    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net()
    original_images = np.random.random((2, 1, 32, 32)).astype(np.float32)
    target_features = np.random.random((2, 10)).astype(np.float32)
    inversion_attack = ImageInversionAttack(net, input_shape=(1, 32, 32), input_bound=(0, 1), loss_weights=[1, 0.2, 5])
    inversion_images = inversion_attack.generate(target_features, iters=10)
    avg_ssim = inversion_attack.evaluate(original_images, inversion_images)
    assert 0 < avg_ssim[1] < 1
    assert target_features.shape[0] == inversion_images.shape[0]


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_inversion_attack_cpu():
    context.set_context(device_target='CPU')
    net = Net()
    original_images = np.random.random((2, 1, 32, 32)).astype(np.float32)
    target_features = np.random.random((2, 10)).astype(np.float32)
    inversion_attack = ImageInversionAttack(net, input_shape=(1, 32, 32), input_bound=(0, 1), loss_weights=[1, 0.2, 5])
    inversion_images = inversion_attack.generate(target_features, iters=10)
    avg_ssim = inversion_attack.evaluate(original_images, inversion_images)
    assert 0 < avg_ssim[1] < 1
    assert target_features.shape[0] == inversion_images.shape[0]


@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
@pytest.mark.component_mindarmour
def test_inversion_attack2():
    net = Net()
    original_images = np.random.random((2, 1, 32, 32)).astype(np.float32)
    target_features = np.random.random((2, 10)).astype(np.float32)
    inversion_attack = ImageInversionAttack(net, input_shape=(1, 32, 32), input_bound=(0, 1), loss_weights=[1, 0.2, 5])
    inversion_images = inversion_attack.generate(target_features, iters=10)
    true_labels = np.array([1, 2])
    new_net = Net()
    indexes = inversion_attack.evaluate(original_images, inversion_images, true_labels, new_net)
    assert len(indexes) == 3

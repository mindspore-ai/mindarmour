# Copyright 2020 Huawei Technologies Co., Ltd
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
Fuction:
    Test fgsm attack about resnet50 network
Usage:
    py.test test_cifar10_attack_fgsm.py
"""
import os
import numpy as np

import pytest

from mindspore import Tensor
from mindspore import context
from mindspore.nn import Cell
from mindspore.common import dtype as mstype
from mindspore.ops import operations as P
from mindspore.ops import functional as F

from mindarmour.attacks.gradient_method import FastGradientSignMethod

from resnet_cifar10 import resnet50_cifar10

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")



class CrossEntropyLoss(Cell):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.cross_entropy = P.SoftmaxCrossEntropyWithLogits()
        self.mean = P.ReduceMean()
        self.one_hot = P.OneHot()
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)

    def construct(self, logits, label):
        label = self.one_hot(label, F.shape(logits)[1], self.on_value, self.off_value)
        loss = self.cross_entropy(logits, label)[0]
        loss = self.mean(loss, (-1,))
        return loss


@pytest.mark.level0
@pytest.mark.env_single
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_ascend_inference
def test_fast_gradient_sign_method():
    """
    FGSM-Attack test
    """
    context.set_context(mode=context.GRAPH_MODE)
    # get network
    net = resnet50_cifar10(10)

    # create test data
    test_images = np.random.rand(64, 3, 224, 224).astype(np.float32)
    test_labels = np.random.randint(10, size=64).astype(np.int32)
    # attacking
    loss_fn = CrossEntropyLoss()
    attack = FastGradientSignMethod(net, eps=0.1, loss_fn=loss_fn)
    adv_data = attack.batch_generate(test_images, test_labels, batch_size=32)
    assert np.any(adv_data != test_images)

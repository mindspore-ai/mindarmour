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
membership inference test
"""
import pytest

import numpy as np

import mindspore.dataset as ds
from mindspore import nn
from mindspore.train import Model
import mindspore.context as context

from mindarmour.privacy.evaluation import MembershipInference

from tests.ut.python.utils.mock_net import Net


context.set_context(mode=context.GRAPH_MODE)


def dataset_generator():
    """mock training data."""
    batch_size = 16
    batches = 1
    data = np.random.randn(batches*batch_size, 1, 32, 32).astype(
        np.float32)
    label = np.random.randint(0, 10, batches*batch_size).astype(np.int32)
    for i in range(batches):
        yield data[i*batch_size:(i + 1)*batch_size],\
              label[i*batch_size:(i + 1)*batch_size]


@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
@pytest.mark.component_mindarmour
def test_get_membership_inference_object():
    net = Net()
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
    opt = nn.Momentum(params=net.trainable_params(), learning_rate=0.1, momentum=0.9)
    model = Model(network=net, loss_fn=loss, optimizer=opt)
    inference_model = MembershipInference(model, -1)
    assert isinstance(inference_model, MembershipInference)


@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.component_mindarmour
def test_membership_inference_object_train():
    net = Net()
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
    opt = nn.Momentum(params=net.trainable_params(), learning_rate=0.1, momentum=0.9)
    model = Model(network=net, loss_fn=loss, optimizer=opt)
    inference_model = MembershipInference(model, 2)
    assert isinstance(inference_model, MembershipInference)

    config = [{
        "method": "KNN",
        "params": {
            "n_neighbors": [3, 5, 7],
        }
    }]
    ds_train = ds.GeneratorDataset(dataset_generator,
                                   ["image", "label"])
    ds_test = ds.GeneratorDataset(dataset_generator,
                                  ["image", "label"])
    inference_model.train(ds_train, ds_test, config)


@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
@pytest.mark.component_mindarmour
def test_membership_inference_eval():
    net = Net()
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
    opt = nn.Momentum(params=net.trainable_params(), learning_rate=0.1, momentum=0.9)
    model = Model(network=net, loss_fn=loss, optimizer=opt)
    inference_model = MembershipInference(model, -1)
    assert isinstance(inference_model, MembershipInference)

    eval_train = ds.GeneratorDataset(dataset_generator,
                                     ["image", "label"])
    eval_test = ds.GeneratorDataset(dataset_generator,
                                    ["image", "label"])

    metrics = ["precision", "accuracy", "recall"]
    inference_model.eval(eval_train, eval_test, metrics)

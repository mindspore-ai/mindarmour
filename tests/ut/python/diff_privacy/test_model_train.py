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
DP-Model test.
"""
import pytest
import numpy as np

from mindspore import nn
from mindspore.nn import SGD
from mindspore.model_zoo.lenet import LeNet5
from mindspore import context
import mindspore.dataset as ds

from mindarmour.diff_privacy import DPOptimizerClassFactory
from mindarmour.diff_privacy import DPModel


def dataset_generator(batch_size, batches):
    data = np.random.random((batches * batch_size, 1, 32, 32)).astype(np.float32)
    label = np.random.randint(0, 10, batches * batch_size).astype(np.int32)
    for i in range(batches):
        yield data[i * batch_size:(i + 1) * batch_size], label[i * batch_size:(i + 1) * batch_size]


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_dp_model():
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    l2_norm_bound = 1.0
    initial_noise_multiplier = 0.01
    net = LeNet5()
    batch_size = 32
    batches = 128
    epochs = 1
    loss = nn.SoftmaxCrossEntropyWithLogits(is_grad=False, sparse=True)
    optim = SGD(params=net.trainable_params(), learning_rate=0.1, momentum=0.9)
    gaussian_mech = DPOptimizerClassFactory()
    gaussian_mech.set_mechanisms('Gaussian',
                                 norm_bound=l2_norm_bound,
                                 initial_noise_multiplier=initial_noise_multiplier)
    model = DPModel(micro_batches=2,
                    norm_clip=l2_norm_bound,
                    dp_mech=gaussian_mech.mech,
                    network=net,
                    loss_fn=loss,
                    optimizer=optim,
                    metrics=None)
    ms_ds = ds.GeneratorDataset(dataset_generator(batch_size, batches), ['data', 'label'])
    ms_ds.set_dataset_size(batch_size * batches)
    model.train(epochs, ms_ds)

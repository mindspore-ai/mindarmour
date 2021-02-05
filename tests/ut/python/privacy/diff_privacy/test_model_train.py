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
from mindspore import context
import mindspore.dataset as ds

from mindarmour.privacy.diff_privacy import DPModel
from mindarmour.privacy.diff_privacy import NoiseMechanismsFactory
from mindarmour.privacy.diff_privacy import ClipMechanismsFactory
from mindarmour.privacy.diff_privacy import DPOptimizerClassFactory

from tests.ut.python.utils.mock_net import Net


def dataset_generator():
    """mock training data."""
    batch_size = 32
    batches = 128
    data = np.random.random((batches*batch_size, 1, 32, 32)).astype(
        np.float32)
    label = np.random.randint(0, 10, batches*batch_size).astype(np.int32)
    for i in range(batches):
        yield data[i*batch_size:(i + 1)*batch_size],\
              label[i*batch_size:(i + 1)*batch_size]


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_dp_model_with_pynative_mode():
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    norm_bound = 1.0
    initial_noise_multiplier = 0.01
    network = Net()
    epochs = 1
    micro_batches = 2
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
    factory_opt = DPOptimizerClassFactory(micro_batches=micro_batches)
    factory_opt.set_mechanisms('Gaussian',
                               norm_bound=norm_bound,
                               initial_noise_multiplier=initial_noise_multiplier)
    net_opt = factory_opt.create('Momentum')(network.trainable_params(),
                                             learning_rate=0.1, momentum=0.9)
    clip_mech = ClipMechanismsFactory().create('Gaussian',
                                               decay_policy='Linear',
                                               learning_rate=0.01,
                                               target_unclipped_quantile=0.9,
                                               fraction_stddev=0.01)
    model = DPModel(micro_batches=micro_batches,
                    norm_bound=norm_bound,
                    clip_mech=clip_mech,
                    noise_mech=None,
                    network=network,
                    loss_fn=loss,
                    optimizer=net_opt,
                    metrics=None)
    ms_ds = ds.GeneratorDataset(dataset_generator,
                                ['data', 'label'])
    model.train(epochs, ms_ds, dataset_sink_mode=False)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_dp_model_with_graph_mode():
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    norm_bound = 1.0
    initial_noise_multiplier = 0.01
    network = Net()
    epochs = 1
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
    noise_mech = NoiseMechanismsFactory().create('Gaussian',
                                                 norm_bound=norm_bound,
                                                 initial_noise_multiplier=initial_noise_multiplier)
    clip_mech = ClipMechanismsFactory().create('Gaussian',
                                               decay_policy='Linear',
                                               learning_rate=0.01,
                                               target_unclipped_quantile=0.9,
                                               fraction_stddev=0.01)
    net_opt = nn.Momentum(network.trainable_params(), learning_rate=0.1,
                          momentum=0.9)
    model = DPModel(micro_batches=2,
                    clip_mech=clip_mech,
                    norm_bound=norm_bound,
                    noise_mech=noise_mech,
                    network=network,
                    loss_fn=loss,
                    optimizer=net_opt,
                    metrics=None)
    ms_ds = ds.GeneratorDataset(dataset_generator,
                                ['data', 'label'])
    model.train(epochs, ms_ds, dataset_sink_mode=False)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_dp_model_with_graph_mode_ada_gaussian():
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    norm_bound = 1.0
    initial_noise_multiplier = 0.01
    network = Net()
    epochs = 1
    alpha = 0.8
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
    noise_mech = NoiseMechanismsFactory().create('AdaGaussian',
                                                 norm_bound=norm_bound,
                                                 initial_noise_multiplier=initial_noise_multiplier,
                                                 noise_decay_rate=alpha,
                                                 decay_policy='Exp')
    clip_mech = None
    net_opt = nn.Momentum(network.trainable_params(), learning_rate=0.1,
                          momentum=0.9)
    model = DPModel(micro_batches=2,
                    clip_mech=clip_mech,
                    norm_bound=norm_bound,
                    noise_mech=noise_mech,
                    network=network,
                    loss_fn=loss,
                    optimizer=net_opt,
                    metrics=None)
    ms_ds = ds.GeneratorDataset(dataset_generator,
                                ['data', 'label'])
    model.train(epochs, ms_ds, dataset_sink_mode=False)

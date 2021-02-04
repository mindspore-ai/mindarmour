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
DP-Monitor test.
"""
import pytest
import numpy as np

import mindspore.nn as nn
import mindspore.dataset as ds
from mindspore.train import Model
import mindspore.context as context

from mindarmour.privacy.diff_privacy import PrivacyMonitorFactory
from mindarmour.utils.logger import LogUtil

from tests.ut.python.utils.mock_net import Net

LOGGER = LogUtil.get_instance()
TAG = 'DP-Monitor Test'


def dataset_generator():
    batch_size = 16
    batches = 128

    data = np.random.random((batches * batch_size, 1, 32, 32)).astype(
        np.float32)
    label = np.random.randint(0, 10, batches * batch_size).astype(np.int32)
    for i in range(batches):
        yield data[i * batch_size: (i + 1) * batch_size], \
              label[i * batch_size: (i + 1) * batch_size]


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_dp_monitor():
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    batch_size = 16
    epochs = 1
    rdp = PrivacyMonitorFactory.create(policy='rdp', num_samples=60000,
                                       batch_size=batch_size,
                                       initial_noise_multiplier=0.4,
                                       noise_decay_rate=6e-3)
    suggest_epoch = rdp.max_epoch_suggest()
    LOGGER.info(TAG, 'The recommended maximum training epochs is: %s',
                suggest_epoch)
    network = Net()
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    net_opt = nn.Momentum(network.trainable_params(), 0.01, 0.9)

    model = Model(network, net_loss, net_opt)

    LOGGER.info(TAG, "============== Starting Training ==============")
    ds1 = ds.GeneratorDataset(dataset_generator,
                              ["data", "label"])
    model.train(epochs, ds1, callbacks=[rdp], dataset_sink_mode=False)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_dp_monitor_gpu():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    batch_size = 16
    epochs = 1
    rdp = PrivacyMonitorFactory.create(policy='rdp', num_samples=60000,
                                       batch_size=batch_size,
                                       initial_noise_multiplier=0.4,
                                       noise_decay_rate=6e-3)
    suggest_epoch = rdp.max_epoch_suggest()
    LOGGER.info(TAG, 'The recommended maximum training epochs is: %s',
                suggest_epoch)
    network = Net()
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    net_opt = nn.Momentum(network.trainable_params(), 0.01, 0.9)

    model = Model(network, net_loss, net_opt)

    LOGGER.info(TAG, "============== Starting Training ==============")
    ds1 = ds.GeneratorDataset(dataset_generator,
                              ["data", "label"])
    model.train(epochs, ds1, callbacks=[rdp], dataset_sink_mode=False)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_dp_monitor_cpu():
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    batch_size = 16
    epochs = 1
    rdp = PrivacyMonitorFactory.create(policy='rdp', num_samples=60000,
                                       batch_size=batch_size,
                                       initial_noise_multiplier=0.4,
                                       noise_decay_rate=6e-3)
    suggest_epoch = rdp.max_epoch_suggest()
    LOGGER.info(TAG, 'The recommended maximum training epochs is: %s',
                suggest_epoch)
    network = Net()
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    net_opt = nn.Momentum(network.trainable_params(), 0.01, 0.9)

    model = Model(network, net_loss, net_opt)

    LOGGER.info(TAG, "============== Starting Training ==============")
    ds1 = ds.GeneratorDataset(dataset_generator,
                              ["data", "label"])
    model.train(epochs, ds1, callbacks=[rdp], dataset_sink_mode=False)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_dp_monitor_zcdp():
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    batch_size = 16
    epochs = 1
    zcdp = PrivacyMonitorFactory.create(policy='zcdp', num_samples=60000,
                                        batch_size=batch_size,
                                        initial_noise_multiplier=0.4,
                                        noise_decay_rate=6e-3)
    suggest_epoch = zcdp.max_epoch_suggest()
    LOGGER.info(TAG, 'The recommended maximum training epochs is: %s',
                suggest_epoch)
    network = Net()
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    net_opt = nn.Momentum(network.trainable_params(), 0.01, 0.9)

    model = Model(network, net_loss, net_opt)

    LOGGER.info(TAG, "============== Starting Training ==============")
    ds1 = ds.GeneratorDataset(dataset_generator,
                              ["data", "label"])
    model.train(epochs, ds1, callbacks=[zcdp], dataset_sink_mode=False)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_dp_monitor_zcdp_gpu():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    batch_size = 16
    epochs = 1
    zcdp = PrivacyMonitorFactory.create(policy='zcdp', num_samples=60000,
                                        batch_size=batch_size,
                                        initial_noise_multiplier=0.4,
                                        noise_decay_rate=6e-3)
    suggest_epoch = zcdp.max_epoch_suggest()
    LOGGER.info(TAG, 'The recommended maximum training epochs is: %s',
                suggest_epoch)
    network = Net()
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    net_opt = nn.Momentum(network.trainable_params(), 0.01, 0.9)

    model = Model(network, net_loss, net_opt)

    LOGGER.info(TAG, "============== Starting Training ==============")
    ds1 = ds.GeneratorDataset(dataset_generator,
                              ["data", "label"])
    model.train(epochs, ds1, callbacks=[zcdp], dataset_sink_mode=False)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_dp_monitor_zcdp_cpu():
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    batch_size = 16
    epochs = 1
    zcdp = PrivacyMonitorFactory.create(policy='zcdp', num_samples=60000,
                                        batch_size=batch_size,
                                        initial_noise_multiplier=0.4,
                                        noise_decay_rate=6e-3)
    suggest_epoch = zcdp.max_epoch_suggest()
    LOGGER.info(TAG, 'The recommended maximum training epochs is: %s',
                suggest_epoch)
    network = Net()
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    net_opt = nn.Momentum(network.trainable_params(), 0.01, 0.9)

    model = Model(network, net_loss, net_opt)

    LOGGER.info(TAG, "============== Starting Training ==============")
    ds1 = ds.GeneratorDataset(dataset_generator,
                              ["data", "label"])
    model.train(epochs, ds1, callbacks=[zcdp], dataset_sink_mode=False)

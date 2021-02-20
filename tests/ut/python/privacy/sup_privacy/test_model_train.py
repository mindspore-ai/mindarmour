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
Suppress Privacy model test.
"""
import pytest
import numpy as np

from mindspore import nn
from mindspore import context
from mindspore.train.callback import ModelCheckpoint
from mindspore.train.callback import CheckpointConfig
from mindspore.train.callback import LossMonitor
from mindspore.nn.metrics import Accuracy
import mindspore.dataset as ds

from mindarmour.privacy.sup_privacy import SuppressModel
from mindarmour.privacy.sup_privacy import SuppressMasker
from mindarmour.privacy.sup_privacy import SuppressPrivacyFactory
from mindarmour.privacy.sup_privacy import MaskLayerDes

from tests.ut.python.utils.mock_net import Net as LeNet5


def dataset_generator():
    """mock training data."""
    batches = 10
    batch_size = 32
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
def test_suppress_model_with_pynative_mode():
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    networks_l5 = LeNet5()
    epochs = 5
    batch_num = 10
    mask_times = 10
    lr = 0.01
    masklayers_lenet5 = []
    masklayers_lenet5.append(MaskLayerDes("conv1.weight", 0, False, False, -1))
    suppress_ctrl_instance = SuppressPrivacyFactory().create(networks_l5,
                                                             masklayers_lenet5,
                                                             policy="local_train",
                                                             end_epoch=epochs,
                                                             batch_num=batch_num,
                                                             start_epoch=1,
                                                             mask_times=mask_times,
                                                             lr=lr,
                                                             sparse_end=0.50,
                                                             sparse_start=0.0)
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    net_opt = nn.SGD(networks_l5.trainable_params(), lr)
    model_instance = SuppressModel(
        network=networks_l5,
        loss_fn=net_loss,
        optimizer=net_opt,
        metrics={"Accuracy": Accuracy()})
    model_instance.link_suppress_ctrl(suppress_ctrl_instance)
    suppress_masker = SuppressMasker(model=model_instance, suppress_ctrl=suppress_ctrl_instance)
    config_ck = CheckpointConfig(save_checkpoint_steps=batch_num, keep_checkpoint_max=10)
    ckpoint_cb = ModelCheckpoint(prefix="checkpoint_lenet",
                                 directory="./trained_ckpt_file/",
                                 config=config_ck)
    ds_train = ds.GeneratorDataset(dataset_generator, ['data', 'label'])

    model_instance.train(epochs, ds_train, callbacks=[ckpoint_cb, LossMonitor(), suppress_masker],
                         dataset_sink_mode=False)

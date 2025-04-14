# Copyright 2024 Huawei Technologies Co., Ltd
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
Train a CIFAR10CNN network model.
"""
import os

import mindspore.nn as nn
from mindspore import context
from mindspore.nn.metrics import Accuracy
from mindspore.train import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from mindarmour.utils.logger import LogUtil

from examples.common.dataset.data_processing import create_dataset_cifar
from examples.common.networks.cifar10cnn.cifar10cnn_net import CIFAR10CNN
LOGGER = LogUtil.get_instance()
LOGGER.set_level('INFO')

TAG = "CIFAR10CNN_train"


def cifar_train(epoch_size, lr, momentum):
    """
    Generate Dataset and Train
    """
    mnist_path = "../../dataset/CIFAR10"
    # ds = create_dataset_cifar(os.path.join(mnist_path, "train"), 32, 32, repeat_num=1)
    ds = create_dataset_cifar(mnist_path, 32, 32, repeat_num=1)

    network = CIFAR10CNN()
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    net_opt = nn.Momentum(network.trainable_params(), lr, momentum)
    config_ck = CheckpointConfig(save_checkpoint_steps=1875,
                                 keep_checkpoint_max=10)
    ckpoint_cb = ModelCheckpoint(prefix="checkpoint_cifar",
                                 directory="./trained_ckpt_file/",
                                 config=config_ck)
    model = Model(network, net_loss, net_opt, metrics={"Accuracy": Accuracy()})

    LOGGER.info(TAG, "============== Starting Training ==============")
    model.train(epoch_size, ds, callbacks=[ckpoint_cb, LossMonitor()],
                dataset_sink_mode=False)

    LOGGER.info(TAG, "============== Starting Testing ==============")
    ckpt_file_name = "trained_ckpt_file/checkpoint_cifar-10_1562.ckpt"
    param_dict = load_checkpoint(ckpt_file_name)
    load_param_into_net(network, param_dict)
    ds_eval = create_dataset_cifar(mnist_path, 32, 32, repeat_num=1, training=False)
    acc = model.eval(ds_eval, dataset_sink_mode=False)
    LOGGER.info(TAG, "============== Accuracy: %s ==============", acc)


if __name__ == '__main__':
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    cifar_train(10, 0.01, 0.9)

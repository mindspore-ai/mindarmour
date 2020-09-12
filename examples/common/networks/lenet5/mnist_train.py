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
import os

import mindspore.nn as nn
from mindspore import context
from mindspore.nn.metrics import Accuracy
from mindspore.train import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from mindarmour.utils.logger import LogUtil

from examples.common.dataset.data_processing import generate_mnist_dataset
from examples.common.networks.lenet5.lenet5_net import LeNet5

LOGGER = LogUtil.get_instance()
LOGGER.set_level('INFO')

TAG = "Lenet5_train"


def mnist_train(epoch_size, batch_size, lr, momentum):
    mnist_path = "../../dataset/MNIST"
    ds = generate_mnist_dataset(os.path.join(mnist_path, "train"),
                                batch_size=batch_size, repeat_size=1)

    network = LeNet5()
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    net_opt = nn.Momentum(network.trainable_params(), lr, momentum)
    config_ck = CheckpointConfig(save_checkpoint_steps=1875,
                                 keep_checkpoint_max=10)
    ckpoint_cb = ModelCheckpoint(prefix="checkpoint_lenet",
                                 directory="./trained_ckpt_file/",
                                 config=config_ck)
    model = Model(network, net_loss, net_opt, metrics={"Accuracy": Accuracy()})

    LOGGER.info(TAG, "============== Starting Training ==============")
    model.train(epoch_size, ds, callbacks=[ckpoint_cb, LossMonitor()],
                dataset_sink_mode=False)

    LOGGER.info(TAG, "============== Starting Testing ==============")
    ckpt_file_name = "trained_ckpt_file/checkpoint_lenet-10_1875.ckpt"
    param_dict = load_checkpoint(ckpt_file_name)
    load_param_into_net(network, param_dict)
    ds_eval = generate_mnist_dataset(os.path.join(mnist_path, "test"),
                                     batch_size=batch_size)
    acc = model.eval(ds_eval, dataset_sink_mode=False)
    LOGGER.info(TAG, "============== Accuracy: %s ==============", acc)


if __name__ == '__main__':
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    mnist_train(10, 32, 0.01, 0.9)

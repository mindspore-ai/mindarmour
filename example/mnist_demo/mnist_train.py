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
# ============================================================================
import os
import sys

import mindspore.nn as nn
from mindspore import context, Tensor
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train import Model
import mindspore.ops.operations as P
from mindspore.nn.metrics import Accuracy
from mindspore.ops import functional as F
from mindspore.common import dtype as mstype

from mindarmour.utils.logger import LogUtil

from lenet5_net import LeNet5

sys.path.append("..")
from data_processing import generate_mnist_dataset
LOGGER = LogUtil.get_instance()
TAG = 'Lenet5_train'


class CrossEntropyLoss(nn.Cell):
    """
    Define loss for network
    """
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


def mnist_train(epoch_size, batch_size, lr, momentum):
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend",
                        enable_mem_reuse=False)

    lr = lr
    momentum = momentum
    epoch_size = epoch_size
    mnist_path = "./MNIST_unzip/"
    ds = generate_mnist_dataset(os.path.join(mnist_path, "train"),
                                batch_size=batch_size, repeat_size=1)

    network = LeNet5()
    network.set_train()
    net_loss = CrossEntropyLoss()
    net_opt = nn.Momentum(network.trainable_params(), lr, momentum)
    config_ck = CheckpointConfig(save_checkpoint_steps=1875, keep_checkpoint_max=10)
    ckpoint_cb = ModelCheckpoint(prefix="checkpoint_lenet", directory='./trained_ckpt_file/', config=config_ck)
    model = Model(network, net_loss, net_opt, metrics={"Accuracy": Accuracy()})

    LOGGER.info(TAG, "============== Starting Training ==============")
    model.train(epoch_size, ds, callbacks=[ckpoint_cb, LossMonitor()], dataset_sink_mode=False) # train

    LOGGER.info(TAG, "============== Starting Testing ==============")
    param_dict = load_checkpoint("trained_ckpt_file/checkpoint_lenet-10_1875.ckpt")
    load_param_into_net(network, param_dict)
    ds_eval = generate_mnist_dataset(os.path.join(mnist_path, "test"), batch_size=batch_size)
    acc = model.eval(ds_eval)
    LOGGER.info(TAG, "============== Accuracy: %s ==============", acc)


if __name__ == '__main__':
    mnist_train(10, 32, 0.001, 0.9)

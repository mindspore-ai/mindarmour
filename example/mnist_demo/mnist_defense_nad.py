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
"""defense example using nad"""
import sys

import numpy as np
from mindspore import Tensor
from mindspore import context
from mindspore import nn
from mindspore.nn import SoftmaxCrossEntropyWithLogits
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from lenet5_net import LeNet5
from mindarmour.attacks import FastGradientSignMethod
from mindarmour.defenses import NaturalAdversarialDefense
from mindarmour.utils.logger import LogUtil

sys.path.append("..")
from data_processing import generate_mnist_dataset


LOGGER = LogUtil.get_instance()
TAG = 'Nad_Example'


def test_nad_method():
    """
    NAD-Defense test for CPU device.
    """
    # 1. load trained network
    ckpt_name = './trained_ckpt_file/checkpoint_lenet-10_1875.ckpt'
    net = LeNet5()
    load_dict = load_checkpoint(ckpt_name)
    load_param_into_net(net, load_dict)

    loss = SoftmaxCrossEntropyWithLogits(is_grad=False, sparse=True)
    opt = nn.Momentum(net.trainable_params(), 0.01, 0.09)

    nad = NaturalAdversarialDefense(net, loss_fn=loss, optimizer=opt,
                                    bounds=(0.0, 1.0), eps=0.3)

    # 2. get test data
    data_list = "./MNIST_unzip/test"
    batch_size = 32
    ds_test = generate_mnist_dataset(data_list, batch_size=batch_size)
    inputs = []
    labels = []
    for data in ds_test.create_tuple_iterator():
        inputs.append(data[0].astype(np.float32))
        labels.append(data[1])
    inputs = np.concatenate(inputs)
    labels = np.concatenate(labels)

    # 3. get accuracy of test data on original model
    net.set_train(False)
    acc_list = []
    batchs = inputs.shape[0] // batch_size
    for i in range(batchs):
        batch_inputs = inputs[i*batch_size : (i + 1)*batch_size]
        batch_labels = labels[i*batch_size : (i + 1)*batch_size]
        logits = net(Tensor(batch_inputs)).asnumpy()
        label_pred = np.argmax(logits, axis=1)
        acc_list.append(np.mean(batch_labels == label_pred))

    LOGGER.debug(TAG, 'accuracy of TEST data on original model is : %s',
                 np.mean(acc_list))

    # 4. get adv of test data
    attack = FastGradientSignMethod(net, eps=0.3, loss_fn=loss)
    adv_data = attack.batch_generate(inputs, labels)
    LOGGER.debug(TAG, 'adv_data.shape is : %s', adv_data.shape)

    # 5. get accuracy of adv data on original model
    net.set_train(False)
    acc_list = []
    batchs = adv_data.shape[0] // batch_size
    for i in range(batchs):
        batch_inputs = adv_data[i*batch_size : (i + 1)*batch_size]
        batch_labels = labels[i*batch_size : (i + 1)*batch_size]
        logits = net(Tensor(batch_inputs)).asnumpy()
        label_pred = np.argmax(logits, axis=1)
        acc_list.append(np.mean(batch_labels == label_pred))

    LOGGER.debug(TAG, 'accuracy of adv data on original model is : %s',
                 np.mean(acc_list))

    # 6. defense
    net.set_train()
    nad.batch_defense(inputs, labels, batch_size=32, epochs=10)

    # 7. get accuracy of test data on defensed model
    net.set_train(False)
    acc_list = []
    batchs = inputs.shape[0] // batch_size
    for i in range(batchs):
        batch_inputs = inputs[i*batch_size : (i + 1)*batch_size]
        batch_labels = labels[i*batch_size : (i + 1)*batch_size]
        logits = net(Tensor(batch_inputs)).asnumpy()
        label_pred = np.argmax(logits, axis=1)
        acc_list.append(np.mean(batch_labels == label_pred))

    LOGGER.debug(TAG, 'accuracy of TEST data on defensed model is : %s',
                 np.mean(acc_list))

    # 8. get accuracy of adv data on defensed model
    acc_list = []
    batchs = adv_data.shape[0] // batch_size
    for i in range(batchs):
        batch_inputs = adv_data[i*batch_size : (i + 1)*batch_size]
        batch_labels = labels[i*batch_size : (i + 1)*batch_size]
        logits = net(Tensor(batch_inputs)).asnumpy()
        label_pred = np.argmax(logits, axis=1)
        acc_list.append(np.mean(batch_labels == label_pred))

    LOGGER.debug(TAG, 'accuracy of adv data on defensed model is : %s',
                 np.mean(acc_list))


if __name__ == '__main__':
    # device_target can be "CPU", "GPU" or "Ascend"
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    test_nad_method()

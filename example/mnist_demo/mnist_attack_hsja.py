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
import sys

import numpy as np
import pytest
from mindspore import Tensor
from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from lenet5_net import LeNet5
from mindarmour.attacks.black.black_model import BlackModel
from mindarmour.attacks.black.hop_skip_jump_attack import HopSkipJumpAttack
from mindarmour.utils.logger import LogUtil

sys.path.append("..")
from data_processing import generate_mnist_dataset

context.set_context(mode=context.GRAPH_MODE)
context.set_context(device_target="Ascend")

LOGGER = LogUtil.get_instance()
TAG = 'HopSkipJumpAttack'


class ModelToBeAttacked(BlackModel):
    """model to be attack"""

    def __init__(self, network):
        super(ModelToBeAttacked, self).__init__()
        self._network = network

    def predict(self, inputs):
        """predict"""
        if len(inputs.shape) == 3:
            inputs = inputs[np.newaxis, :]
        result = self._network(Tensor(inputs.astype(np.float32)))
        return result.asnumpy()


def random_target_labels(true_labels):
    target_labels = []
    for label in true_labels:
        while True:
            target_label = np.random.randint(0, 10)
            if target_label != label:
                target_labels.append(target_label)
                break
    return target_labels


def create_target_images(dataset, data_labels, target_labels):
    res = []
    for label in target_labels:
        for data_label, data in zip(data_labels, dataset):
            if data_label == label:
                res.append(data)
                break
    return np.array(res)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_hsja_mnist_attack():
    """
    hsja-Attack test
    """
    # upload trained network
    ckpt_name = './trained_ckpt_file/checkpoint_lenet-10_1875.ckpt'
    net = LeNet5()
    load_dict = load_checkpoint(ckpt_name)
    load_param_into_net(net, load_dict)
    net.set_train(False)

    # get test data
    data_list = "./MNIST_unzip/test"
    batch_size = 32
    ds = generate_mnist_dataset(data_list, batch_size=batch_size)

    # prediction accuracy before attack
    model = ModelToBeAttacked(net)
    batch_num = 5  # the number of batches of attacking samples
    test_images = []
    test_labels = []
    predict_labels = []
    i = 0
    for data in ds.create_tuple_iterator():
        i += 1
        images = data[0].astype(np.float32)
        labels = data[1]
        test_images.append(images)
        test_labels.append(labels)
        pred_labels = np.argmax(model.predict(images), axis=1)
        predict_labels.append(pred_labels)
        if i >= batch_num:
            break
    predict_labels = np.concatenate(predict_labels)
    true_labels = np.concatenate(test_labels)
    accuracy = np.mean(np.equal(predict_labels, true_labels))
    LOGGER.info(TAG, "prediction accuracy before attacking is : %s",
                accuracy)
    test_images = np.concatenate(test_images)

    # attacking
    norm = 'l2'
    search = 'grid_search'
    target = False
    attack = HopSkipJumpAttack(model, constraint=norm, stepsize_search=search)
    if target:
        target_labels = random_target_labels(true_labels)
        target_images = create_target_images(test_images, predict_labels,
                                             target_labels)
        attack.set_target_images(target_images)
        success_list, adv_data, _ = attack.generate(test_images, target_labels)
    else:
        success_list, adv_data, _ = attack.generate(test_images, None)

    adv_datas = []
    gts = []
    for success, adv, gt in zip(success_list, adv_data, true_labels):
        if success:
            adv_datas.append(adv)
            gts.append(gt)
    if gts:
        adv_datas = np.concatenate(np.asarray(adv_datas), axis=0)
        gts = np.asarray(gts)
        pred_logits_adv = model.predict(adv_datas)
        pred_lables_adv = np.argmax(pred_logits_adv, axis=1)
        accuracy_adv = np.mean(np.equal(pred_lables_adv, gts))
        LOGGER.info(TAG, 'mis-classification rate of adversaries is : %s',
                    accuracy_adv)


if __name__ == '__main__':
    test_hsja_mnist_attack()

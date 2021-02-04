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
import os

import numpy as np
import pytest
from mindspore import Tensor
from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from mindarmour import BlackModel
from mindarmour.adv_robustness.attacks import HopSkipJumpAttack
from mindarmour.utils.logger import LogUtil

from tests.ut.python.utils.mock_net import Net

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
        for i, data_label in enumerate(data_labels):
            if data_label == label:
                res.append(dataset[i])
                break
    return np.array(res)


# public variable
def get_model():
    # upload trained network
    current_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_path = os.path.join(current_dir,
                             '../../../dataset/trained_ckpt_file/checkpoint_lenet-10_1875.ckpt')
    net = Net()
    load_dict = load_checkpoint(ckpt_path)
    load_param_into_net(net, load_dict)
    net.set_train(False)
    model = ModelToBeAttacked(net)
    return model


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_hsja_mnist_attack():
    """
    hsja-Attack test
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))


    # get test data
    test_images_set = np.load(os.path.join(current_dir,
                                           '../../../dataset/test_images.npy'))
    test_labels_set = np.load(os.path.join(current_dir,
                                           '../../../dataset/test_labels.npy'))
    # prediction accuracy before attack
    model = get_model()
    batch_num = 1  # the number of batches of attacking samples
    predict_labels = []
    i = 0

    for img in test_images_set:
        i += 1
        pred_labels = np.argmax(model.predict(img), axis=1)
        predict_labels.append(pred_labels)
        if i >= batch_num:
            break
    predict_labels = np.concatenate(predict_labels)
    true_labels = test_labels_set[:batch_num]
    accuracy = np.mean(np.equal(predict_labels, true_labels))
    LOGGER.info(TAG, "prediction accuracy before attacking is : %s",
                accuracy)
    test_images = test_images_set[:batch_num]

    # attacking
    norm = 'l2'
    search = 'grid_search'
    target = False

    attack = HopSkipJumpAttack(model, constraint=norm, stepsize_search=search)
    if target:
        target_labels = random_target_labels(true_labels)
        target_images = create_target_images(test_images_set, test_labels_set,
                                             target_labels)
        LOGGER.info(TAG, 'len target labels : %s', len(target_labels))
        LOGGER.info(TAG, 'len target_images : %s', len(target_images))
        LOGGER.info(TAG, 'len test_images : %s', len(test_images))
        attack.set_target_images(target_images)
        success_list, adv_data, _ = attack.generate(test_images, target_labels)
    else:
        success_list, adv_data, _ = attack.generate(test_images, None)
    assert (adv_data != test_images).any()

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


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_value_error():
    model = get_model()
    norm = 'l2'
    with pytest.raises(ValueError):
        assert HopSkipJumpAttack(model, constraint=norm, stepsize_search='bad-search')

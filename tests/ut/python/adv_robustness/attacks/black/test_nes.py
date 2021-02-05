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
from mindarmour.adv_robustness.attacks import NES
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


def _pseudorandom_target(index, total_indices, true_class):
    """ pseudo random_target """
    rng = np.random.RandomState(index)
    target = true_class
    while target == true_class:
        target = rng.randint(0, total_indices)
    return target


def create_target_images(dataset, data_labels, target_labels):
    res = []
    for label in target_labels:
        for i, data_label in enumerate(data_labels):
            if data_label == label:
                res.append(dataset[i])
                break
    return np.array(res)


def get_model(current_dir):
    ckpt_path = os.path.join(current_dir,
                             '../../../dataset/trained_ckpt_file/checkpoint_lenet-10_1875.ckpt')
    net = Net()
    load_dict = load_checkpoint(ckpt_path)
    load_param_into_net(net, load_dict)
    net.set_train(False)
    model = ModelToBeAttacked(net)
    return model


def get_dataset(current_dir):
    # upload trained network

    # get test data
    test_images = np.load(os.path.join(current_dir,
                                       '../../../dataset/test_images.npy'))
    test_labels = np.load(os.path.join(current_dir,
                                       '../../../dataset/test_labels.npy'))
    return test_images, test_labels


def nes_mnist_attack(scene, top_k):
    """
    hsja-Attack test
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    test_images, test_labels = get_dataset(current_dir)
    model = get_model(current_dir)
    # prediction accuracy before attack
    batch_num = 5  # the number of batches of attacking samples
    predict_labels = []
    i = 0
    for img in test_images:
        i += 1
        pred_labels = np.argmax(model.predict(img), axis=1)
        predict_labels.append(pred_labels)
        if i >= batch_num:
            break
    predict_labels = np.concatenate(predict_labels)
    true_labels = test_labels
    accuracy = np.mean(np.equal(predict_labels, true_labels[:batch_num]))
    LOGGER.info(TAG, "prediction accuracy before attacking is : %s",
                accuracy)
    test_images = test_images

    # attacking
    if scene == 'Query_Limit':
        top_k = -1
    elif scene == 'Partial_Info':
        top_k = top_k
    elif scene == 'Label_Only':
        top_k = top_k

    success = 0
    queries_num = 0

    nes_instance = NES(model, scene, top_k=top_k)
    test_length = 1
    advs = []
    for img_index in range(test_length):
        # INITIAL IMAGE AND CLASS SELECTION
        initial_img = test_images[img_index]
        orig_class = true_labels[img_index]
        initial_img = [initial_img]
        target_class = random_target_labels([orig_class])
        target_image = create_target_images(test_images, true_labels,
                                            target_class)

        nes_instance.set_target_images(target_image)
        tag, adv, queries = nes_instance.generate(np.array(initial_img), np.array(target_class))
        if tag[0]:
            success += 1
        queries_num += queries[0]
        advs.append(adv)

    advs = np.reshape(advs, (len(advs), 1, 32, 32))
    assert (advs != test_images[:batch_num]).any()

    adv_pred = np.argmax(model.predict(advs), axis=1)
    _ = np.mean(np.equal(adv_pred, true_labels[:test_length]))


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_nes_query_limit():
    # scene is in ['Query_Limit', 'Partial_Info', 'Label_Only']
    scene = 'Query_Limit'
    nes_mnist_attack(scene, top_k=-1)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_nes_partial_info():
    # scene is in ['Query_Limit', 'Partial_Info', 'Label_Only']
    scene = 'Partial_Info'
    nes_mnist_attack(scene, top_k=5)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_nes_label_only():
    # scene is in ['Query_Limit', 'Partial_Info', 'Label_Only']
    scene = 'Label_Only'
    nes_mnist_attack(scene, top_k=5)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_value_error():
    """test that exception is raised for invalid labels"""
    with pytest.raises(ValueError):
        assert nes_mnist_attack('Label_Only', -1)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_none():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model = get_model(current_dir)
    test_images, test_labels = get_dataset(current_dir)
    nes = NES(model, 'Partial_Info')
    with pytest.raises(ValueError):
        assert nes.generate(test_images, test_labels)

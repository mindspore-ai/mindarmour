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
"""evaluate example"""
import os
import time

import numpy as np
from mindspore import Model
from mindspore import Tensor
from mindspore import context
from mindspore import nn
from mindspore.nn import Cell
from mindspore.nn import SoftmaxCrossEntropyWithLogits
from mindspore.ops.operations import Add
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from scipy.special import softmax

from mindarmour.adv_robustness.attacks import FastGradientSignMethod
from mindarmour.adv_robustness.attacks import GeneticAttack
from mindarmour.adv_robustness.attacks.black.black_model import BlackModel
from mindarmour.adv_robustness.defenses import NaturalAdversarialDefense
from mindarmour.adv_robustness.detectors import SimilarityDetector
from mindarmour.adv_robustness.evaluations import BlackDefenseEvaluate
from mindarmour.adv_robustness.evaluations import DefenseEvaluate
from mindarmour.utils.logger import LogUtil

from examples.common.dataset.data_processing import generate_mnist_dataset
from examples.common.networks.lenet5.lenet5_net import LeNet5

LOGGER = LogUtil.get_instance()
LOGGER.set_level('INFO')
TAG = 'Defense_Evaluate_Example'


def get_detector(train_images):
    encoder = Model(EncoderNet(encode_dim=256))
    detector = SimilarityDetector(max_k_neighbor=50, trans_model=encoder)
    detector.fit(inputs=train_images)
    return detector


class EncoderNet(Cell):
    """
    Similarity encoder for input data
    """

    def __init__(self, encode_dim):
        super(EncoderNet, self).__init__()
        self._encode_dim = encode_dim
        self.add = Add()

    def construct(self, inputs):
        """
        construct the neural network
        Args:
            inputs (Tensor): input data to neural network.
        Returns:
            Tensor, output of neural network.
        """
        return self.add(inputs, inputs)

    def get_encode_dim(self):
        """
        Get the dimension of encoded inputs

        Returns:
            int, dimension of encoded inputs.
        """
        return self._encode_dim


class ModelToBeAttacked(BlackModel):
    """
    model to be attack
    """

    def __init__(self, network, defense=False, train_images=None):
        super(ModelToBeAttacked, self).__init__()
        self._network = network
        self._queries = []
        self._defense = defense
        self._detector = None
        self._detected_res = []
        if self._defense:
            self._detector = get_detector(train_images)

    def predict(self, inputs):
        """
        predict function
        """
        if len(inputs.shape) == 3:
            inputs = np.expand_dims(inputs, axis=0)
        query_num = inputs.shape[0]
        results = []
        if self._detector:
            for i in range(query_num):
                query = np.expand_dims(inputs[i].astype(np.float32), axis=0)
                result = self._network(Tensor(query)).asnumpy()
                det_num = len(self._detector.get_detected_queries())
                self._detector.detect(np.array([query]))
                new_det_num = len(self._detector.get_detected_queries())
                # If attack query detected, return random predict result
                if new_det_num > det_num:
                    results.append(result + np.random.rand(*result.shape))
                    self._detected_res.append(True)
                else:
                    results.append(result)
                    self._detected_res.append(False)
            results = np.concatenate(results)
        else:
            if len(inputs.shape) == 3:
                inputs = np.expand_dims(inputs, axis=0)
            results = self._network(Tensor(inputs.astype(np.float32))).asnumpy()
        return results

    def get_detected_result(self):
        return self._detected_res


def test_defense_evaluation():
    # load trained network
    current_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_path = os.path.abspath(os.path.join(
        current_dir, '../../common/networks/lenet5/trained_ckpt_file/checkpoint_lenet-10_1875.ckpt'))
    wb_net = LeNet5()
    load_dict = load_checkpoint(ckpt_path)
    load_param_into_net(wb_net, load_dict)

    # get test data
    data_list = "../../common/dataset/MNIST/test"
    batch_size = 32
    ds_test = generate_mnist_dataset(data_list, batch_size=batch_size)
    inputs = []
    labels = []
    for data in ds_test.create_tuple_iterator(output_numpy=True):
        inputs.append(data[0].astype(np.float32))
        labels.append(data[1])
    inputs = np.concatenate(inputs).astype(np.float32)
    labels = np.concatenate(labels).astype(np.int32)

    target_label = np.random.randint(0, 10, size=labels.shape[0])
    for idx in range(labels.shape[0]):
        while target_label[idx] == labels[idx]:
            target_label[idx] = np.random.randint(0, 10)
    target_label = np.eye(10)[target_label].astype(np.float32)

    attacked_size = 50
    benign_size = 500

    attacked_sample = inputs[:attacked_size]
    attacked_true_label = labels[:attacked_size]
    benign_sample = inputs[attacked_size:attacked_size + benign_size]

    wb_model = ModelToBeAttacked(wb_net)

    # gen white-box adversarial examples of test data
    loss = SoftmaxCrossEntropyWithLogits(sparse=True)
    wb_attack = FastGradientSignMethod(wb_net, eps=0.3, loss_fn=loss)
    wb_adv_sample = wb_attack.generate(attacked_sample,
                                       attacked_true_label)

    wb_raw_preds = softmax(wb_model.predict(wb_adv_sample), axis=1)
    accuracy_test = np.mean(
        np.equal(np.argmax(wb_model.predict(attacked_sample), axis=1),
                 attacked_true_label))
    LOGGER.info(TAG, "prediction accuracy before white-box attack is : %s",
                accuracy_test)
    accuracy_adv = np.mean(np.equal(np.argmax(wb_raw_preds, axis=1),
                                    attacked_true_label))
    LOGGER.info(TAG, "prediction accuracy after white-box attack is : %s",
                accuracy_adv)

    # improve the robustness of model with white-box adversarial examples
    opt = nn.Momentum(wb_net.trainable_params(), 0.01, 0.09)

    nad = NaturalAdversarialDefense(wb_net, loss_fn=loss, optimizer=opt,
                                    bounds=(0.0, 1.0), eps=0.3)
    wb_net.set_train(False)
    nad.batch_defense(inputs[:5000], labels[:5000], batch_size=32, epochs=10)

    wb_def_preds = wb_net(Tensor(wb_adv_sample)).asnumpy()
    wb_def_preds = softmax(wb_def_preds, axis=1)
    accuracy_def = np.mean(np.equal(np.argmax(wb_def_preds, axis=1),
                                    attacked_true_label))
    LOGGER.info(TAG, "prediction accuracy after defense is : %s", accuracy_def)

    # calculate defense evaluation metrics for defense against white-box attack
    wb_def_evaluate = DefenseEvaluate(wb_raw_preds, wb_def_preds,
                                      attacked_true_label)
    LOGGER.info(TAG, 'defense evaluation for white-box adversarial attack')
    LOGGER.info(TAG,
                'classification accuracy variance (CAV) is : {:.2f}'.format(
                    wb_def_evaluate.cav()))
    LOGGER.info(TAG, 'classification rectify ratio (CRR) is : {:.2f}'.format(
        wb_def_evaluate.crr()))
    LOGGER.info(TAG, 'classification sacrifice ratio (CSR) is : {:.2f}'.format(
        wb_def_evaluate.csr()))
    LOGGER.info(TAG,
                'classification confidence variance (CCV) is : {:.2f}'.format(
                    wb_def_evaluate.ccv()))
    LOGGER.info(TAG, 'classification output stability is : {:.2f}'.format(
        wb_def_evaluate.cos()))

    # calculate defense evaluation metrics for defense against black-box attack
    LOGGER.info(TAG, 'defense evaluation for black-box adversarial attack')
    bb_raw_preds = []
    bb_def_preds = []
    raw_query_counts = []
    raw_query_time = []
    def_query_counts = []
    def_query_time = []
    def_detection_counts = []

    # gen black-box adversarial examples of test data
    bb_net = LeNet5()
    load_param_into_net(bb_net, load_dict)
    bb_model = ModelToBeAttacked(bb_net, defense=False)
    attack_rm = GeneticAttack(model=bb_model, pop_size=6, mutation_rate=0.05,
                              per_bounds=0.5, step_size=0.25, temp=0.1,
                              sparse=False)
    attack_target_label = target_label[:attacked_size]
    true_label = labels[:attacked_size + benign_size]
    # evaluate robustness of original model
    # gen black-box adversarial examples of test data
    for idx in range(attacked_size):
        raw_st = time.time()
        _, raw_a, raw_qc = attack_rm.generate(
            np.expand_dims(attacked_sample[idx], axis=0),
            np.expand_dims(attack_target_label[idx], axis=0))
        raw_t = time.time() - raw_st
        bb_raw_preds.extend(softmax(bb_model.predict(raw_a), axis=1))
        raw_query_counts.extend(raw_qc)
        raw_query_time.append(raw_t)

    for idx in range(benign_size):
        raw_st = time.time()
        bb_raw_pred = softmax(
            bb_model.predict(np.expand_dims(benign_sample[idx], axis=0)),
            axis=1)
        raw_t = time.time() - raw_st
        bb_raw_preds.extend(bb_raw_pred)
        raw_query_counts.extend([0])
        raw_query_time.append(raw_t)

    accuracy_test = np.mean(
        np.equal(np.argmax(bb_raw_preds[0:len(attack_target_label)], axis=1),
                 np.argmax(attack_target_label, axis=1)))
    LOGGER.info(TAG, "attack success before adv defense is : %s",
                accuracy_test)

    # improve the robustness of model with similarity-based detector
    bb_def_model = ModelToBeAttacked(bb_net, defense=True,
                                     train_images=inputs[0:6000])
    # attack defensed model
    attack_dm = GeneticAttack(model=bb_def_model, pop_size=6,
                              mutation_rate=0.05,
                              per_bounds=0.5, step_size=0.25, temp=0.1,
                              sparse=False)
    for idx in range(attacked_size):
        def_st = time.time()
        _, def_a, def_qc = attack_dm.generate(
            np.expand_dims(attacked_sample[idx], axis=0),
            np.expand_dims(attack_target_label[idx], axis=0))
        def_t = time.time() - def_st
        det_res = bb_def_model.get_detected_result()
        def_detection_counts.append(np.sum(det_res[-def_qc[0]:]))
        bb_def_preds.extend(softmax(bb_def_model.predict(def_a), axis=1))
        def_query_counts.extend(def_qc)
        def_query_time.append(def_t)

    for idx in range(benign_size):
        def_st = time.time()
        bb_def_pred = softmax(
            bb_def_model.predict(np.expand_dims(benign_sample[idx], axis=0)),
            axis=1)
        def_t = time.time() - def_st
        det_res = bb_def_model.get_detected_result()
        def_detection_counts.append(np.sum(det_res[-1]))
        bb_def_preds.extend(bb_def_pred)
        def_query_counts.extend([0])
        def_query_time.append(def_t)

    accuracy_adv = np.mean(
        np.equal(np.argmax(bb_def_preds[0:len(attack_target_label)], axis=1),
                 np.argmax(attack_target_label, axis=1)))
    LOGGER.info(TAG, "attack success rate after adv defense is : %s",
                accuracy_adv)

    bb_raw_preds = np.array(bb_raw_preds).astype(np.float32)
    bb_def_preds = np.array(bb_def_preds).astype(np.float32)
    # check evaluate data
    max_queries = 6000

    def_evaluate = BlackDefenseEvaluate(bb_raw_preds, bb_def_preds,
                                        np.array(raw_query_counts),
                                        np.array(def_query_counts),
                                        np.array(raw_query_time),
                                        np.array(def_query_time),
                                        np.array(def_detection_counts),
                                        true_label, max_queries)

    LOGGER.info(TAG, 'query count variance of adversaries is : {:.2f}'.format(
        def_evaluate.qcv()))
    LOGGER.info(TAG, 'attack success rate variance of adversaries '
                     'is : {:.2f}'.format(def_evaluate.asv()))
    LOGGER.info(TAG, 'false positive rate (FPR) of the query-based detector '
                     'is : {:.2f}'.format(def_evaluate.fpr()))
    LOGGER.info(TAG, 'the benign query response time variance (QRV) '
                     'is : {:.2f}'.format(def_evaluate.qrv()))


if __name__ == '__main__':
    # device_target can be "CPU", "GPU" or "Ascend"
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    DEVICE = context.get_context("device_target")
    if DEVICE in ("Ascend", "GPU"):
        test_defense_evaluation()

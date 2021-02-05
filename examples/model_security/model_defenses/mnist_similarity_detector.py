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
import numpy as np
from scipy.special import softmax

from mindspore import Model
from mindspore import Tensor
from mindspore import context
from mindspore.nn import Cell
from mindspore.ops.operations import Add
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from mindarmour import BlackModel
from mindarmour.adv_robustness.attacks.black.pso_attack import PSOAttack
from mindarmour.adv_robustness.detectors import SimilarityDetector
from mindarmour.utils.logger import LogUtil

from examples.common.dataset.data_processing import generate_mnist_dataset
from examples.common.networks.lenet5.lenet5_net import LeNet5

LOGGER = LogUtil.get_instance()
LOGGER.set_level('INFO')
TAG = 'Similarity Detector test'


class ModelToBeAttacked(BlackModel):
    """
    model to be attack
    """

    def __init__(self, network):
        super(ModelToBeAttacked, self).__init__()
        self._network = network
        self._queries = []

    def predict(self, inputs):
        """
        predict function
        """
        query_num = inputs.shape[0]
        for i in range(query_num):
            if len(inputs[i].shape) == 2:
                temp = np.expand_dims(inputs[i], axis=0)
            else:
                temp = inputs[i]
            self._queries.append(temp.astype(np.float32))
        if len(inputs.shape) == 3:
            inputs = np.expand_dims(inputs, axis=0)
        result = self._network(Tensor(inputs.astype(np.float32)))
        return result.asnumpy()

    def get_queries(self):
        return self._queries


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


def test_similarity_detector():
    """
    Similarity Detector test.
    """
    # load trained network
    ckpt_path = '../../common/networks/lenet5/trained_ckpt_file/checkpoint_lenet-10_1875.ckpt'
    net = LeNet5()
    load_dict = load_checkpoint(ckpt_path)
    load_param_into_net(net, load_dict)

    # get mnist data
    data_list = "../../common/dataset/MNIST/test"
    batch_size = 1000
    ds = generate_mnist_dataset(data_list, batch_size=batch_size)
    model = ModelToBeAttacked(net)

    batch_num = 10  # the number of batches of input samples
    all_images = []
    true_labels = []
    predict_labels = []
    i = 0
    for data in ds.create_tuple_iterator(output_numpy=True):
        i += 1
        images = data[0].astype(np.float32)
        labels = data[1]
        all_images.append(images)
        true_labels.append(labels)
        pred_labels = np.argmax(model.predict(images), axis=1)
        predict_labels.append(pred_labels)
        if i >= batch_num:
            break
    all_images = np.concatenate(all_images)
    true_labels = np.concatenate(true_labels)
    predict_labels = np.concatenate(predict_labels)
    accuracy = np.mean(np.equal(predict_labels, true_labels))
    LOGGER.info(TAG, "prediction accuracy before attacking is : %s", accuracy)

    train_images = all_images[0:6000, :, :, :]
    attacked_images = all_images[0:10, :, :, :]
    attacked_labels = true_labels[0:10]

    # generate malicious query sequence of black attack
    attack = PSOAttack(model, bounds=(0.0, 1.0), pm=0.5, sparse=True,
                       t_max=1000)
    success_list, adv_data, query_list = attack.generate(attacked_images,
                                                         attacked_labels)
    LOGGER.info(TAG, 'pso attack success_list: %s', success_list)
    LOGGER.info(TAG, 'average of query counts is : %s', np.mean(query_list))
    pred_logits_adv = model.predict(adv_data)
    # rescale predict confidences into (0, 1).
    pred_logits_adv = softmax(pred_logits_adv, axis=1)
    pred_lables_adv = np.argmax(pred_logits_adv, axis=1)
    accuracy_adv = np.mean(np.equal(pred_lables_adv, attacked_labels))
    LOGGER.info(TAG, "prediction accuracy after attacking is : %g",
                accuracy_adv)

    benign_queries = all_images[6000:10000, :, :, :]
    suspicious_queries = model.get_queries()

    # explicit threshold not provided, calculate threshold for K
    encoder = Model(EncoderNet(encode_dim=256))
    detector = SimilarityDetector(max_k_neighbor=50, trans_model=encoder)
    detector.fit(inputs=train_images)

    # test benign queries
    detector.detect(benign_queries)
    fpr = len(detector.get_detected_queries()) / benign_queries.shape[0]
    LOGGER.info(TAG, 'Number of false positive of attack detector is : %s',
                len(detector.get_detected_queries()))
    LOGGER.info(TAG, 'False positive rate of attack detector is : %s', fpr)

    # test attack queries
    detector.clear_buffer()
    detector.detect(np.array(suspicious_queries))
    LOGGER.info(TAG, 'Number of detected attack queries is : %s',
                len(detector.get_detected_queries()))
    LOGGER.info(TAG, 'The detected attack query indexes are : %s',
                detector.get_detected_queries())


if __name__ == '__main__':
    # device_target can be "CPU", "GPU" or "Ascend"
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    DEVICE = context.get_context("device_target")
    if DEVICE in ("Ascend", "GPU"):
        test_similarity_detector()

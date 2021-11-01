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

import numpy as np
from mindspore import Tensor
from mindspore.train.model import Model
from mindspore import Model, nn, context
from examples.common.networks.lenet5.lenet5_net_for_fuzzing import LeNet5
from mindspore.train.summary.summary_record import _get_summary_tensor_data
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindarmour.reliability.concept_drift.concept_drift_check_images import OodDetector, result_eval


"""
Examples for Lenet.
"""


def feature_extract(data, feature_model, layer='output[:Tensor]'):
    """
    Extract features.
    Args:
        data (numpy.ndarray): Input data.
        feature_model (Model): The model for extracting features.
        layer (str): The feature layer. The layer name could be 'output[:Tensor]',
                    '1[:Tensor]', '2[:Tensor]',...'10[:Tensor]'.

    Returns:
        numpy.ndarray, the feature of input data.
    """
    feature_model.predict(Tensor(data))
    layer_out = _get_summary_tensor_data()
    return layer_out[layer].asnumpy()


if __name__ == '__main__':
    # load model
    ckpt_path = '../../tests/ut/python/dataset/trained_ckpt_file/checkpoint_lenet-10_1875.ckpt'
    net = LeNet5()
    load_dict = load_checkpoint(ckpt_path)
    load_param_into_net(net, load_dict)
    model = Model(net)
    # load data
    ds_train = np.load('../../tests/ut/python/dataset/concept_train_lenet.npy')
    ds_test = np.load('../../tests/ut/python/dataset/concept_test_lenet.npy')
    ds_train = feature_extract(ds_train, model, layer='output[:Tensor]')
    ds_test = feature_extract(ds_test, model, layer='output[:Tensor]')
    # ood detect
    detector = OodDetector(ds_train, ds_test, n_cluster=10)
    score = detector.ood_detector()
    # Evaluation
    num = int(len(ds_test)/2)
    label = np.concatenate((np.zeros(num), np.ones(num)), axis=0)  # ID data = 0, OOD data = 1
    dec_acc = result_eval(score, label, threshold=0.5)

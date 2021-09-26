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
from examples.common.networks.resnet.resnet import resnet50
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindarmour.reliability.concept_drift.concept_drift_check_images import OodDetectorFeatureCluster


"""
Examples for Resnet.
"""


if __name__ == '__main__':
    # load model
    ckpt_path = '../../tests/ut/python/dataset/trained_ckpt_file/resnet_1-20_1875.ckpt'
    net = resnet50()
    load_dict = load_checkpoint(ckpt_path)
    load_param_into_net(net, load_dict)
    model = Model(net)
    # load data
    ds_train = np.load('train.npy')
    ds_test1 = np.load('test1.npy')
    ds_test2 = np.load('test2.npy')
    # ood detector initialization
    detector = OodDetectorFeatureCluster(model, ds_train, n_cluster=10, layer='output[:Tensor]')
    # get optimal threshold with ds_test1
    num = int(len(ds_test1) / 2)
    label = np.concatenate((np.zeros(num), np.ones(num)), axis=0)  # ID data = 0, OOD data = 1
    optimal_threshold = detector.get_optimal_threshold(label, ds_test1)
    # get result of ds_test2. We can also set threshold by ourself.
    result = detector.ood_predict(optimal_threshold, ds_test2)

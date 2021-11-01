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
# ============================================================================


import heapq
import numpy as np
from sklearn.cluster import KMeans
from mindarmour.utils._check_param import check_param_type, check_param_in_range

"""
Out-of-Distribution detection for images.
The sample can be run on Ascend 910 AI processor.
"""


class OodDetector:
    """
    Train the OOD detector.

    Args:
        ds_train (numpy.ndarray): The training dataset.
        ds_test (numpy.ndarray): The testing dataset.
    """

    def __init__(self, ds_train, ds_test, n_cluster=10):
        self.ds_train = check_param_type('ds_train', ds_train, np.ndarray)
        self.ds_test = check_param_type('ds_test', ds_test, np.ndarray)
        self.n_cluster = check_param_type('n_cluster', n_cluster, int)
        self.n_cluster = check_param_in_range('n_cluster', n_cluster, 2, 100)

    def ood_detector(self):
        """
        The out-of-distribution detection.

        Returns:
           - numpy.ndarray, the detection score of images.
        """

        clf = KMeans(n_clusters=self.n_cluster)
        clf.fit_predict(self.ds_train)
        feature_cluster = clf.cluster_centers_
        score = []
        for i in range(len(self.ds_test)):
            dis = []
            for j in range(len(feature_cluster)):
                loc = list(map(list(feature_cluster[j]).index, heapq.nlargest(self.n_cluster, list(feature_cluster[j]))))
                diff = sum(abs((feature_cluster[j][loc] - self.ds_test[i][loc]))) / sum(abs((feature_cluster[j][loc])))
                dis.append(diff)
            score.append(min(dis))
        score = np.array(score)
        return score


def result_eval(score, label, threshold):
    """
    Evaluate the detection results.

    Args:
        score (numpy.ndarray): The detection score of images.
        label (numpy.ndarray): The label whether an image is in-ditribution and out-of-distribution.
        threshold (float): The threshold to judge out-of-distribution distance.

    Returns:
        - float, the detection accuracy.
    """
    check_param_type('label', label, np.ndarray)
    check_param_type('threshold', threshold, float)
    check_param_in_range('threshold', threshold, 0, 1)
    count = 0
    for i in range(len(score)):
        if score[i] < threshold and label[i] == 0:
            count = count + 1
        if score[i] >= threshold and label[i] == 1:
            count = count + 1
    return count / len(score)

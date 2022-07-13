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
"""
Out-of-Distribution detection module for images.
"""

import heapq
import numpy as np
from sklearn.cluster import KMeans
from mindspore import Tensor
from mindspore.train.summary.summary_record import _get_summary_tensor_data
from mindarmour.utils._check_param import check_param_type, check_param_in_range


class OodDetector:
    """
    The abstract class of the out-of-distribution detector.

    Args:
        model (Model):The training model.
        ds_train (numpy.ndarray): The training dataset.
    """

    def __init__(self, model, ds_train):
        self.model = model
        self.ds_train = check_param_type('ds_train', ds_train, np.ndarray)

    def _feature_extract(self, model, data, layer='output[:Tensor]'):
        """
        Extract features.
        Args:
            model (Model): The model for extracting features.
            data (numpy.ndarray): Input data.
            layer (str): The name of the feature layer. layer (str) is represented as
                'name[:Tensor]', where 'name' is given by users when training the model.
                Please see more details about how to name the model layer in 'README.md'.

        Returns:
            numpy.ndarray, the data feature extracted by a certain neural layer.
        """
        model.predict(Tensor(data))
        layer_out = _get_summary_tensor_data()
        return layer_out[layer].asnumpy()

    def get_optimal_threshold(self, label, ds_eval):
        """
        Get the optimal threshold. Try to find an optimal threshold value to
        detect OOD examples. The optimal threshold is calculated by a labeled
        dateset `ds_eval`.

        Args:
            label (numpy.ndarray): The label whether an image is in-distribution and out-of-distribution.
            ds_eval (numpy.ndarray): The testing dataset to help find the threshold.

        Returns:
            - float, the optimal threshold.
        """

    def ood_predict(self, threshold, ds_test):
        """
        The out-of-distribution detection. This function aims to detect whether images,
        regarded as `ds_test`, are OOD examples or not. If the prediction score of one
        image is larger than `threshold`, this image is out-of-distribution.

        Args:
            threshold (float): the threshold to judge ood data. One can set value by experience
                or use function get_optimal_threshold.
            ds_test (numpy.ndarray): The testing dataset.

        Returns:
           - numpy.ndarray, the detection result. 0 means the data is not ood, 1 means the data is ood.
        """


class OodDetectorFeatureCluster(OodDetector):
    """
    Train the OOD detector. Extract the training data features, and obtain the clustering centers. The distance between
    the testing data features and the clustering centers determines whether an image is an out-of-distribution(OOD)
    image or not.

    For details, please check `Implementing the Concept Drift Detection Application of Image Data
    <https://mindspore.cn/mindarmour/docs/en/r1.8/concept_drift_images.html>`_.

    Args:
        model (Model):The training model.
        ds_train (numpy.ndarray): The training dataset.
        n_cluster (int): The cluster number. Belonging to [2,100].
            Usually, n_cluster equals to the class number of the training dataset.
            If the OOD detector performs poor in the testing dataset, we can increase the value of n_cluster
            appropriately.
        layer (str): The name of the feature layer. layer (str) is represented by
            'name[:Tensor]', where 'name' is given by users when training the model.
            Please see more details about how to name the model layer in 'README.md'.

    Examples:
        >>> from mindspore import Model
        >>> from mindspore.ops import TensorSummary
        >>> import mindspore.ops.operations as P
        >>> from mindarmour.reliability.concept_drift.concept_drift_check_images import OodDetectorFeatureCluster
        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super(Net, self).__init__()
        ...         self._softmax = P.Softmax()
        ...         self._Dense = nn.Dense(10,10)
        ...         self._squeeze = P.Squeeze(1)
        ...         self._summary = TensorSummary()
        ...     def construct(self, inputs):
        ...         out = self._softmax(inputs)
        ...         out = self._Dense(out)
        ...         self._summary('output', out)
        ...         return self._squeeze(out)
        >>> net = Net()
        >>> model = Model(net)
        >>> batch_size = 16
        >>> batches = 1
        >>> ds_train = np.random.randn(batches * batch_size, 1, 10).astype(np.float32)
        >>> ds_eval = np.random.randn(batches * batch_size, 1, 10).astype(np.float32)
        >>> detector = OodDetectorFeatureCluster(model, ds_train, n_cluster=10, layer='output[:Tensor]')
        >>> num = int(len(ds_eval) / 2)
        >>> ood_label = np.concatenate((np.zeros(num), np.ones(num)), axis=0)
        >>> optimal_threshold = detector.get_optimal_threshold(ood_label, ds_eval)
    """

    def __init__(self, model, ds_train, n_cluster, layer):
        super(OodDetectorFeatureCluster, self).__init__(model, ds_train)
        self.model = model
        self.ds_train = check_param_type('ds_train', ds_train, np.ndarray)
        self.n_cluster = check_param_type('n_cluster', n_cluster, int)
        self.n_cluster = check_param_in_range('n_cluster', n_cluster, 2, 100)
        self.layer = check_param_type('layer', layer, str)
        self.feature = self._feature_extract(model, ds_train, layer=self.layer)

    def _feature_cluster(self):
        """
        Get the feature cluster.

        Returns:
            - numpy.ndarray, the feature cluster.
        """
        clf = KMeans(n_clusters=self.n_cluster)
        clf.fit_predict(self.feature)
        feature_cluster = clf.cluster_centers_
        return feature_cluster

    def _get_ood_score(self, ds_test):
        """
        Get the ood score.

        Args:
            ds_test (numpy.ndarray): The testing dataset.

        Returns:
            - float, the optimal threshold.
        """
        feature_cluster = self._feature_cluster()
        ds_test = self._feature_extract(self.model, ds_test, layer=self.layer)
        score = []
        for i in range(len(ds_test)):
            dis = []
            for j in range(len(feature_cluster)):
                loc = list(
                    map(list(feature_cluster[j]).index, heapq.nlargest(self.n_cluster, list(feature_cluster[j]))))
                diff = sum(abs((feature_cluster[j][loc] - ds_test[i][loc]))) / sum(abs((feature_cluster[j][loc])))
                dis.append(diff)
            score.append(min(dis))
        score = np.array(score)
        return score

    def get_optimal_threshold(self, label, ds_eval):
        """
        Get the optimal threshold. Try to find an optimal threshold value to
        detect OOD examples. The optimal threshold is calculated by a labeled
        dateset `ds_eval`.

        Args:
            label (numpy.ndarray): The label whether an image is in-distribution and out-of-distribution.
            ds_eval (numpy.ndarray): The testing dataset to help find the threshold.

        Returns:
            - float, the optimal threshold.
        """
        check_param_type('label', label, np.ndarray)
        check_param_type('ds_eval', ds_eval, np.ndarray)
        score = self._get_ood_score(ds_eval)
        acc = []
        threshold = []
        for threshold_change in np.arange(0.0, 1.0, 0.01):
            count = 0
            for i in range(len(score)):
                if score[i] < threshold_change and label[i] == 0:
                    count = count + 1
                if score[i] >= threshold_change and label[i] == 1:
                    count = count + 1
            acc.append(count / len(score))
            threshold.append(threshold_change)
        acc = np.array(acc)
        threshold = np.array(threshold)
        optimal_threshold = threshold[np.where(acc == np.max(acc))[0]][0]
        return optimal_threshold

    def ood_predict(self, threshold, ds_test):
        """
        The out-of-distribution detection. This function aims to detect whether images,
        regarded as `ds_test`, are OOD examples or not. If the prediction score of one
        image is larger than `threshold`, this image is out-of-distribution.

        Args:
            threshold (float): the threshold to judge ood data. One can set value by experience
                or use function get_optimal_threshold.
            ds_test (numpy.ndarray): The testing dataset.

        Returns:
           - numpy.ndarray, the detection result. 0 means the data is not ood, 1 means the data is ood.
        """
        score = self._get_ood_score(ds_test)
        result = []
        for i in range(len(score)):
            if score[i] < threshold:
                result.append(0)
            if score[i] >= threshold:
                result.append(1)
        result = np.array(result)
        return result

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
"""
Similarity Detector.
"""
import itertools
import numpy as np

from mindspore import Tensor
from mindspore import Model

from mindarmour.utils.logger import LogUtil
from mindarmour.utils._check_param import check_model, check_numpy_param, \
    check_int_positive, check_value_positive, check_param_type, \
    check_param_in_range
from ..detector import Detector

LOGGER = LogUtil.get_instance()
TAG = 'SimilarityDetector'


def _pairwise_distances(x_input, y_input):
    """
    Compute the Euclidean Distance matrix from a vector array x_input and
    y_input.

    Args:
        x_input (numpy.ndarray): input data, [n_samples_x, n_features]
        y_input (numpy.ndarray): input data, [n_samples_y, n_features]

    Returns:
        numpy.ndarray, distance matrix, [n_samples_a, n_samples_b]
    """
    out = np.empty((x_input.shape[0], y_input.shape[0]), dtype='float')
    iterator = itertools.product(
        range(x_input.shape[0]), range(y_input.shape[0]))
    for i, j in iterator:
        out[i, j] = np.linalg.norm(x_input[i] - y_input[j])
    return out


class SimilarityDetector(Detector):
    """
    The detector measures similarity among adjacent queries and rejects queries
    which are remarkably similar to previous queries.

    Reference: `Stateful Detection of Black-Box Adversarial Attacks by Steven
    Chen, Nicholas Carlini, and David Wagner. at arxiv 2019
    <https://arxiv.org/abs/1907.05587>`_

    Args:
        trans_model (Model): A MindSpore model to encode input data into lower
            dimension vector.
        max_k_neighbor (int): The maximum number of the nearest neighbors.
            Default: 1000.
        chunk_size (int): Buffer size. Default: 1000.
        max_buffer_size (int): Maximum buffer size. Default: 10000.
        tuning (bool): Calculate the average distance for the nearest k
            neighbours, if tuning is true, k=K. If False k=1,...,K.
            Default: False.
        fpr (float): False positive ratio on legitimate query sequences.
            Default: 0.001

    Examples:
        >>> detector = SimilarityDetector(model)
        >>> detector.fit(ori, labels)
        >>> adv_ids = detector.detect(adv)
    """

    def __init__(self, trans_model, max_k_neighbor=1000, chunk_size=1000,
                 max_buffer_size=10000, tuning=False, fpr=0.001):
        super(SimilarityDetector, self).__init__()
        self._max_k_neighbor = check_int_positive('max_k_neighbor',
                                                  max_k_neighbor)
        self._trans_model = check_model('trans_model', trans_model, Model)
        self._tuning = check_param_type('tuning', tuning, bool)
        self._chunk_size = check_int_positive('chunk_size', chunk_size)
        self._max_buffer_size = check_int_positive('max_buffer_size',
                                                   max_buffer_size)
        self._fpr = check_param_in_range('fpr', fpr, 0, 1)
        self._num_of_neighbors = None
        self._threshold = None
        self._num_queries = 0
        # Stores recently processed queries
        self._buffer = []
        # Tracks indexes of detected queries
        self._detected_queries = []

    def fit(self, inputs, labels=None):
        """
        Process input training data to calculate the threshold.
        A proper threshold should make sure the false positive
        rate is under a given value.

        Args:
            inputs (numpy.ndarray): Training data to calculate the threshold.
            labels (numpy.ndarray): Labels of training data.

        Returns:
            - list[int], number of the nearest neighbors.

            - list[float], calculated thresholds for different K.

        Raises:
            ValueError: The number of training data is less than
                max_k_neighbor!
        """
        data = check_numpy_param('inputs', inputs)
        data_len = data.shape[0]
        if data_len < self._max_k_neighbor:
            raise ValueError('The number of training data must be larger than '
                             'max_k_neighbor!')
        data = self._trans_model.predict(Tensor(data)).asnumpy()
        data = data.reshape((data.shape[0], -1))
        distances = []
        for i in range(data.shape[0] // self._chunk_size):
            distance_mat = _pairwise_distances(
                x_input=data[i*self._chunk_size:(i + 1)*self._chunk_size, :],
                y_input=data)
            distance_mat = np.sort(distance_mat, axis=-1)
            distances.append(distance_mat[:, :self._max_k_neighbor])
        # the rest
        distance_mat = _pairwise_distances(x_input=data[(data.shape[0] //
                                                         self._chunk_size)*
                                                        self._chunk_size:, :],
                                           y_input=data)
        distance_mat = np.sort(distance_mat, axis=-1)
        distances.append(distance_mat[:, :self._max_k_neighbor])

        distance_matrix = np.concatenate(distances, axis=0)

        start = 1 if self._tuning else self._max_k_neighbor

        thresholds = []
        num_nearest_neighbors = []
        for k in range(start, self._max_k_neighbor + 1):
            avg_dist = distance_matrix[:, :k].mean(axis=-1)
            index = int(len(avg_dist)*self._fpr)
            threshold = np.sort(avg_dist, axis=None)[index]
            num_nearest_neighbors.append(k)
            thresholds.append(threshold)
        if thresholds:
            self._threshold = thresholds[-1]
            self._num_of_neighbors = num_nearest_neighbors[-1]
        return num_nearest_neighbors, thresholds

    def detect(self, inputs):
        """
        Process queries to detect black-box attack.

        Args:
             inputs (numpy.ndarray): Query sequence.

        Raises:
            ValueError: The parameters of threshold or num_of_neighbors is
                not available.
        """
        if self._threshold is None or self._num_of_neighbors is None:
            msg = 'Explicit detection threshold and number of nearest ' \
                  'neighbors must be provided using set_threshold(), ' \
                  'or call fit() to calculate.'
            LOGGER.error(TAG, msg)
            raise ValueError(msg)
        queries = check_numpy_param('inputs', inputs)
        queries = self._trans_model.predict(Tensor(queries)).asnumpy()
        queries = queries.reshape((queries.shape[0], -1))
        for query in queries:
            self._process_query(query)

    def _process_query(self, query):
        """
        Process each query to detect black-box attack.

        Args:
             query (numpy.ndarray): Query input.
        """
        if len(self._buffer) < self._num_of_neighbors:
            self._buffer.append(query)
            self._num_queries += 1
            return
        k = self._num_of_neighbors

        if self._buffer:
            queries = np.stack(self._buffer, axis=0)
            dists = np.linalg.norm(queries - query, axis=-1)

        k_nearest_dists = np.partition(dists, k - 1)[:k, None]
        k_avg_dist = np.mean(k_nearest_dists)

        self._buffer.append(query)
        self._num_queries += 1

        if len(self._buffer) >= self._max_buffer_size:
            self.clear_buffer()

        # an attack is detected
        if k_avg_dist < self._threshold:
            self._detected_queries.append(self._num_queries)
            self.clear_buffer()

    def clear_buffer(self):
        """
        Clear the buffer memory.

        """
        while self._buffer:
            self._buffer.pop()

    def set_threshold(self, num_of_neighbors, threshold):
        """
        Set the parameters num_of_neighbors and threshold.

        Args:
            num_of_neighbors (int): Number of the nearest neighbors.
            threshold (float): Detection threshold.
        """
        self._num_of_neighbors = check_int_positive('num_of_neighbors',
                                                    num_of_neighbors)
        self._threshold = check_value_positive('threshold', threshold)

    def get_detection_interval(self):
        """
        Get the interval between adjacent detections.

        Returns:
            list[int], number of queries between adjacent detections.
        """
        detected_queries = self._detected_queries
        interval = []
        for i in range(len(detected_queries) - 1):
            interval.append(detected_queries[i + 1] - detected_queries[i])
        return interval

    def get_detected_queries(self):
        """
        Get the indexes of detected queries.

        Returns:
            list[int], sequence number of detected malicious queries.
        """
        detected_queries = self._detected_queries
        return detected_queries

    def detect_diff(self, inputs):
        """
        Detect adversarial samples from input samples, like the predict_proba
        function in common machine learning model.

        Args:
            inputs (Union[numpy.ndarray, list, tuple]): Data been used as
                references to create adversarial examples.

        Raises:
            NotImplementedError: This function is not available
                in class `SimilarityDetector`.
        """
        msg = 'The function detect_diff() is not available in the class ' \
              '`SimilarityDetector`.'
        LOGGER.error(TAG, msg)
        raise NotImplementedError(msg)

    def transform(self, inputs):
        """
        Filter adversarial noises in input samples.

        Raises:
            NotImplementedError: This function is not available
                in class `SimilarityDetector`.
        """
        msg = 'The function transform() is not available in the class ' \
              '`SimilarityDetector`.'
        LOGGER.error(TAG, msg)
        raise NotImplementedError(msg)

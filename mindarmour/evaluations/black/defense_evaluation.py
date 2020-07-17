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
Evaluating Defense against Black-box Attacks.
"""
import numpy as np

from mindarmour.utils.logger import LogUtil
from mindarmour.utils._check_param import check_pair_numpy_param, \
    check_equal_length, check_int_positive, check_numpy_param

LOGGER = LogUtil.get_instance()
TAG = 'BlackDefenseEvaluate'


class BlackDefenseEvaluate:
    """
    Evaluation metrics of anti-black-box defense method.

    Args:
        raw_preds (numpy.ndarray): Predict results of some certain samples on
            raw model.
        def_preds (numpy.ndarray): Predict results of some certain samples on
            defensed model.
        raw_query_counts (numpy.ndarray): Number of queries to generate
            adversarial examples on raw model, which is one dimensional whose
            size is raw_preds.shape[0]. For benign samples, query count must be
            set to 0.
        def_query_counts (numpy.ndarray): Number of queries to generate
            adversarial examples on defensed model, which is one dimensional
            whose size is raw_preds.shape[0].
            For benign samples, query count must be set to 0.
        raw_query_time (numpy.ndarray): The total time duration to generate
            an adversarial example on raw model, which is one dimensional
            whose size is raw_preds.shape[0].
        def_query_time (numpy.ndarray): The total time duration to generate an
            adversarial example on defensed model, which is one dimensional
            whose size is raw_preds.shape[0].
        def_detection_counts (numpy.ndarray): Total number of detected queries
            during each adversarial example generation, which is one dimensional
            whose size is raw_preds.shape[0]. For a benign sample, the
            def_detection_counts is set to 1 if the query is identified as
            suspicious, and 0 otherwise.
        true_labels (numpy.ndarray): True labels in one-dim whose size is
            raw_preds.shape[0].
        max_queries (int): Attack budget, the maximum number of queries.

    Examples:
        >>> raw_preds = np.array([[0.1, 0.1, 0.2, 0.6],
        >>>                     [0.1, 0.7, 0.0, 0.2],
        >>>                     [0.8, 0.1, 0.0, 0.1]])
        >>> def_preds = np.array([[0.1, 0.1, 0.1, 0.7],
        >>>                     [0.1, 0.6, 0.2, 0.1],
        >>>                     [0.1, 0.2, 0.1, 0.6]])
        >>> raw_query_counts = np.array([0,20,10])
        >>> def_query_counts = np.array([0,50,60])
        >>> raw_query_time = np.array([0.1, 2, 1])
        >>> def_query_time = np.array([0.2, 6, 5])
        >>> def_detection_counts = np.array([1, 5, 10])
        >>> true_labels = np.array([3, 1, 0])
        >>> max_queries = 100
        >>> def_eval = BlackDefenseEvaluate(raw_preds,
        >>>                             def_preds,
        >>>                             raw_query_counts,
        >>>                             def_query_counts,
        >>>                             raw_query_time,
        >>>                             def_query_time,
        >>>                             def_detection_counts,
        >>>                             true_labels,
        >>>                             max_queries)
        >>> def_eval.qcv()
    """

    def __init__(self, raw_preds, def_preds, raw_query_counts, def_query_counts,
                 raw_query_time, def_query_time, def_detection_counts,
                 true_labels, max_queries):
        self._raw_preds, self._def_preds = check_pair_numpy_param('raw_preds',
                                                                  raw_preds,
                                                                  'def_preds',
                                                                  def_preds)
        self._num_samples = self._raw_preds.shape[0]
        self._raw_query_counts, _ = check_equal_length('raw_query_counts',
                                                       raw_query_counts,
                                                       'number of sample',
                                                       self._raw_preds)
        self._def_query_counts, _ = check_equal_length('def_query_counts',
                                                       def_query_counts,
                                                       'number of sample',
                                                       self._raw_preds)
        self._raw_query_time, _ = check_equal_length('raw_query_time',
                                                     raw_query_time,
                                                     'number of sample',
                                                     self._raw_preds)
        self._def_query_time, _ = check_equal_length('def_query_time',
                                                     def_query_time,
                                                     'number of sample',
                                                     self._raw_preds)

        self._num_adv_samples = self._raw_query_counts[
            self._raw_query_counts > 0].shape[0]

        self._num_adv_samples = check_int_positive(
            'the number of adversarial samples',
            self._num_adv_samples)

        self._num_ben_samples = self._num_samples - self._num_adv_samples
        self._max_queries = check_int_positive('max_queries', max_queries)

        self._def_detection_counts = check_numpy_param('def_detection_counts',
                                                       def_detection_counts)
        self._true_labels = check_numpy_param('true_labels', true_labels)

    def qcv(self):
        """
        Calculate query count variance (QCV).

        Returns:
            float, the higher, the stronger the defense is. If num_adv_samples=0,
            return -1.
        """
        if self._num_adv_samples == 0:
            return -1
        avg_def_query_count = \
            np.sum(self._def_query_counts) / self._num_adv_samples
        avg_raw_query_count = \
            np.sum(self._raw_query_counts) / self._num_adv_samples

        if (avg_def_query_count == self._max_queries) \
                and (avg_raw_query_count < self._max_queries):
            query_variance = 1
        else:
            query_variance = \
                min(avg_def_query_count - avg_raw_query_count,
                    self._max_queries) / self._max_queries
        return query_variance

    def asv(self):
        """
        Calculate attack success rate variance (ASV).

        Returns:
            float, the lower, the stronger the defense is. If num_adv_samples=0,
            return -1.
        """
        adv_def_preds = self._def_preds[self._def_query_counts > 0]
        adv_raw_preds = self._raw_preds[self._raw_query_counts > 0]
        adv_true_labels = self._true_labels[self._raw_query_counts > 0]

        def_succ_num = np.sum(np.argmax(adv_def_preds, axis=1)
                              != adv_true_labels)
        raw_succ_num = np.sum(np.argmax(adv_raw_preds, axis=1)
                              != adv_true_labels)
        if self._num_adv_samples == 0:
            return -1
        return (raw_succ_num - def_succ_num) / self._num_adv_samples

    def fpr(self):
        """
        Calculate false positive rate (FPR) of the query-based detector.

        Returns:
            float, the lower, the higher usability the defense is. If
            num_adv_samples=0, return -1.
        """

        ben_detect_counts = \
            self._def_detection_counts[self._def_query_counts == 0]
        num_fp = ben_detect_counts[ben_detect_counts > 0].shape[0]
        if self._num_ben_samples == 0:
            return -1
        return num_fp / self._num_ben_samples

    def qrv(self):
        """
        Calculate the benign query response time variance (QRV).

        Returns:
            float, the lower, the higher usability the defense is. If
            num_adv_samples=0, return -1.
        """
        if self._num_ben_samples == 0:
            return -1
        raw_num_queries = self._num_ben_samples
        def_num_queries = self._num_ben_samples

        ben_raw_query_time = self._raw_query_time[self._raw_query_counts == 0]
        ben_def_query_time = self._def_query_time[self._def_query_counts == 0]

        avg_raw_query_time = np.sum(ben_raw_query_time) / raw_num_queries
        avg_def_query_time = np.sum(ben_def_query_time) / def_num_queries

        return (avg_def_query_time -
                avg_raw_query_time) / (avg_raw_query_time + 1e-12)

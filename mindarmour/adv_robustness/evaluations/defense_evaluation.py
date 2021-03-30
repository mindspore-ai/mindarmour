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
Defense Evaluation.
"""
import numpy as np

import scipy.stats as st

from mindarmour.utils.logger import LogUtil
from mindarmour.utils._check_param import check_numpy_param
from mindarmour.utils._check_param import check_pair_numpy_param

LOGGER = LogUtil.get_instance()
TAG = 'DefenseEvaluate'


class DefenseEvaluate:
    """
    Evaluation metrics of defense methods.

    Args:
        raw_preds (numpy.ndarray): Prediction results of some certain samples
            on raw model.
        def_preds (numpy.ndarray): Prediction results of some certain samples on
            defensed model.
        true_labels (numpy.ndarray): Ground-truth labels of samples, a
            one-dimension array whose size is raw_preds.shape[0].

    Examples:
        >>> raw_preds = np.array([[0.1, 0.1, 0.2, 0.6],
        >>>                       [0.1, 0.7, 0.0, 0.2],
        >>>                       [0.8, 0.1, 0.0, 0.1]])
        >>> def_preds = np.array([[0.1, 0.1, 0.1, 0.7],
        >>>                       [0.1, 0.6, 0.2, 0.1],
        >>>                       [0.1, 0.2, 0.1, 0.6]])
        >>> true_labels = np.array([3, 1, 0])
        >>> def_eval = DefenseEvaluate(raw_preds,
        >>>                            def_preds,
        >>>                            true_labels)
        >>> def_eval.cav()
    """
    def __init__(self, raw_preds, def_preds, true_labels):
        self._raw_preds, self._def_preds = check_pair_numpy_param('raw_preds',
                                                                  raw_preds,
                                                                  'def_preds',
                                                                  def_preds)
        self._true_labels = check_numpy_param('true_labels', true_labels)
        self._num_samples = len(true_labels)

    def cav(self):
        """
        Calculate classification accuracy variance (CAV).

        Returns:
            float, the higher, the more successful the defense is.
        """
        def_succ_num = np.sum(np.argmax(self._def_preds, axis=1)
                              == self._true_labels)
        raw_succ_num = np.sum(np.argmax(self._raw_preds, axis=1)
                              == self._true_labels)

        return (def_succ_num - raw_succ_num) / self._num_samples

    def crr(self):
        """
        Calculate classification rectify ratio (CRR).

        Returns:
            float, the higher, the more successful the defense is.
        """
        cond1 = np.argmax(self._def_preds, axis=1) == self._true_labels
        cond2 = np.argmax(self._raw_preds, axis=1) != self._true_labels
        rectify_num = np.sum(cond1*cond2)

        return rectify_num*1.0 / self._num_samples

    def csr(self):
        """
        Calculate classification sacrifice ratio (CSR), the lower the better.

        Returns:
            float, the lower, the more successful the defense is.
        """
        cond1 = np.argmax(self._def_preds, axis=1) != self._true_labels
        cond2 = np.argmax(self._raw_preds, axis=1) == self._true_labels
        sacrifice_num = np.sum(cond1*cond2)

        return sacrifice_num*1.0 / self._num_samples

    def ccv(self):
        """
        Calculate classification confidence variance (CCV).

        Returns:
            - float, the lower, the more successful the defense is.

              - If return value == -1, len(idxes) == 0.
        """
        idxes = np.arange(self._num_samples)
        cond1 = np.argmax(self._def_preds, axis=1) == self._true_labels
        cond2 = np.argmax(self._raw_preds, axis=1) == self._true_labels
        idxes = idxes[cond1*cond2]

        def_max = np.max(self._def_preds, axis=1)
        raw_max = np.max(self._raw_preds, axis=1)

        if idxes.shape[0] == 0:
            return -1
        conf_variance = np.mean(np.abs(def_max[idxes] - raw_max[idxes]))

        return conf_variance

    def cos(self):
        """
        References: `Calculate classification output stability (COS)
        <https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence>`_

        Returns:
            float.
                - If return value >= 0, is effective defense. The lower, the
                  more successful the defense.

                - If return value == -1, idxes == 0.
        """
        idxes = np.arange(self._num_samples)
        cond1 = np.argmax(self._def_preds, axis=1) == self._true_labels
        cond2 = np.argmax(self._raw_preds, axis=1) == self._true_labels
        idxes = idxes[cond1*cond2]
        if idxes.size == 0:
            return -1
        def_preds = self._def_preds[idxes]
        raw_preds = self._raw_preds[idxes]

        js_total = 0.0
        mean_value = 0.5*(def_preds + raw_preds)
        for i, value in enumerate(mean_value):
            js_total += 0.5*st.entropy(def_preds[i], value) \
                        + 0.5*st.entropy(raw_preds[i], value)

        return js_total / len(idxes)

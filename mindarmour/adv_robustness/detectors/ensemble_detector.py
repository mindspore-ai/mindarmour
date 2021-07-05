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
Ensemble Detector.
"""
import numpy as np

from mindarmour.utils.logger import LogUtil
from mindarmour.utils._check_param import check_numpy_param, \
    check_param_multi_types
from .detector import Detector

LOGGER = LogUtil.get_instance()
TAG = 'EnsembleDetector'


class EnsembleDetector(Detector):
    """
    Ensemble detector.

    Args:
        detectors (Union[tuple, list]): List of detector methods.
        policy (str): Decision policy, could be 'vote', 'all' or 'any'.
            Default: 'vote'
    """

    def __init__(self, detectors, policy="vote"):
        super(EnsembleDetector, self).__init__()
        self._detectors = check_param_multi_types('detectors', detectors,
                                                  [list, tuple])
        self._num_detectors = len(detectors)
        self._policy = policy

    def fit(self, inputs, labels=None):
        """
        Fit detector like a machine learning model. This method is not available
        in this class.

        Args:
            inputs (numpy.ndarray): Data to calculate the threshold.
            labels (numpy.ndarray): Labels of data. Default: None.

        Raises:
            NotImplementedError: This function is not available in ensemble.
        """
        msg = 'The function fit() is not available in the class ' \
              '`EnsembleDetector`.'
        LOGGER.error(TAG, msg)
        raise NotImplementedError(msg)

    def detect(self, inputs):
        """
        Detect adversarial examples from input samples.

        Args:
            inputs (numpy.ndarray): Input samples.

        Returns:
            list[int], whether a sample is adversarial. if res[i]=1, then the
            input sample with index i is adversarial.

        Raises:
            ValueError: If policy is not supported.
        """

        inputs = check_numpy_param('inputs', inputs)
        x_len = inputs.shape[0]
        counts = np.zeros(x_len)
        res = np.zeros(x_len, dtype=np.int)
        for detector in list(self._detectors):
            idx = detector.detect(inputs)
            counts[idx] += 1

        if self._policy == "vote":
            idx_adv = np.argwhere(counts > self._num_detectors / 2)
        elif self._policy == "all":
            idx_adv = np.argwhere(counts == self._num_detectors)
        elif self._policy == "any":
            idx_adv = np.argwhere(counts > 0)
        else:
            msg = 'Policy {} is not supported.'.format(self._policy)
            LOGGER.error(TAG, msg)
            raise ValueError(msg)
        res[idx_adv] = 1
        return list(res)

    def detect_diff(self, inputs):
        """
        This method is not available in this class.

        Args:
            inputs (Union[numpy.ndarray, list, tuple]): Data been used as
                references to create adversarial examples.

        Raises:
            NotImplementedError: This function is not available in ensemble.
        """
        msg = 'The function detect_diff() is not available in the class ' \
              '`EnsembleDetector`.'
        LOGGER.error(TAG, msg)
        raise NotImplementedError(msg)

    def transform(self, inputs):
        """
        Filter adversarial noises in input samples.
        This method is not available in this class.

        Args:
            inputs (Union[numpy.ndarray, list, tuple]): Data been used as
                references to create adversarial examples.

        Raises:
            NotImplementedError: This function is not available in ensemble.
        """
        msg = 'The function transform() is not available in the class ' \
              '`EnsembleDetector`.'
        LOGGER.error(TAG, msg)
        raise NotImplementedError(msg)

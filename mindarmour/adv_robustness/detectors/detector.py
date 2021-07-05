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
Base Class of Detector.
"""
from abc import abstractmethod

from mindarmour.utils.logger import LogUtil

LOGGER = LogUtil.get_instance()
TAG = 'Detector'


class Detector:
    """
    The abstract base class for all adversarial example detectors.
    """
    def __init__(self):
        pass


    @abstractmethod
    def fit(self, inputs, labels=None):
        """
        Fit a threshold and refuse adversarial examples whose difference from
        their denoised versions are larger than the threshold. The threshold is
        determined by a certain false positive rate when applying to normal samples.

        Args:
            inputs (numpy.ndarray): The input samples to calculate the threshold.
            labels (numpy.ndarray): Labels of training data. Default: None.

        Raises:
            NotImplementedError: It is an abstract method.
        """
        msg = 'The function fit() is an abstract function in class ' \
              '`Detector` and should be implemented in child class.'
        LOGGER.error(TAG, msg)
        raise NotImplementedError(msg)

    @abstractmethod
    def detect(self, inputs):
        """
        Detect adversarial examples from input samples.

        Args:
            inputs (Union[numpy.ndarray, list, tuple]): The input samples to be
                detected.

        Raises:
            NotImplementedError: It is an abstract method.
        """
        msg = 'The function detect() is an abstract function in class ' \
              '`Detector` and should be implemented in child class.'
        LOGGER.error(TAG, msg)
        raise NotImplementedError(msg)

    @abstractmethod
    def detect_diff(self, inputs):
        """
        Calculate the difference between the input samples and de-noised samples.

        Args:
            inputs (Union[numpy.ndarray, list, tuple]): The input samples to be
                detected.

        Raises:
            NotImplementedError: It is an abstract method.

        """
        msg = 'The function detect_diff() is an abstract function in class ' \
              '`Detector` and should be implemented in child class.'
        LOGGER.error(TAG, msg)
        raise NotImplementedError(msg)

    @abstractmethod
    def transform(self, inputs):
        """
        Filter adversarial noises in input samples.

        Args:
            inputs (Union[numpy.ndarray, list, tuple]): The input samples to be
                transformed.
        Raises:
            NotImplementedError: It is an abstract method.
        """
        msg = 'The function transform() is an abstract function in class ' \
              '`Detector` and should be implemented in child class.'
        LOGGER.error(TAG, msg)
        raise NotImplementedError(msg)

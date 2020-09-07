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
Black model.
"""
from abc import abstractmethod

import numpy as np

from mindarmour.utils.logger import LogUtil

LOGGER = LogUtil.get_instance()
TAG = 'BlackModel'


class BlackModel:
    """
    The abstract class which treats the target model as a black box. The model
    should be defined by users.
    """
    def __init__(self):
        pass

    @abstractmethod
    def predict(self, inputs):
        """
        Predict using the user specified model. The shape of predict results
        should be (m, n), where n represents the number of classes this model
        classifies.

        Args:
            inputs (numpy.ndarray): The input samples to be predicted.

        Raises:
            NotImplementedError: It is an abstract method.
        """
        msg = 'The function predict() is an abstract function in class ' \
              '`BlackModel` and should be implemented in child class by user.'
        LOGGER.error(TAG, msg)
        raise NotImplementedError(msg)

    def is_adversarial(self, data, label, is_targeted):
        """
        Check if input sample is adversarial example or not.

        Args:
            data (numpy.ndarray): The input sample to be check, typically some
                maliciously perturbed examples.
            label (numpy.ndarray): For targeted attacks, label is intended
                label of perturbed example. For untargeted attacks, label is
                original label of corresponding unperturbed sample.
            is_targeted (bool): For targeted/untargeted attacks, select True/False.

        Returns:
            bool.
                - If True, the input sample is adversarial.

                - If False, the input sample is not adversarial.
        """
        logits = self.predict(np.expand_dims(data, axis=0))[0]
        predicts = np.argmax(logits)
        if is_targeted:
            return predicts == label
        return predicts != label

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
Base Class of Attack.
"""
from abc import abstractmethod

import numpy as np

from mindarmour.utils._check_param import check_pair_numpy_param, \
    check_int_positive
from mindarmour.utils.logger import LogUtil

LOGGER = LogUtil.get_instance()
TAG = 'Attack'


class Attack:
    """
    The abstract base class for all attack classes creating adversarial examples.
    """
    def __init__(self):
        pass

    def batch_generate(self, inputs, labels, batch_size=64):
        """
        Generate adversarial examples in batch, based on input samples and
        their labels.

        Args:
            inputs (numpy.ndarray): Samples based on which adversarial
                examples are generated.
            labels (numpy.ndarray): Labels of samples, whose values determined
                by specific attacks.
            batch_size (int): The number of samples in one batch.

        Returns:
            numpy.ndarray, generated adversarial examples

        Examples:
            >>> inputs = Tensor([[0.2, 0.4, 0.5, 0.2], [0.7, 0.2, 0.4, 0.3]])
            >>> labels = [3, 0]
            >>> advs = attack.batch_generate(inputs, labels, batch_size=2)
        """
        arr_x, arr_y = check_pair_numpy_param('inputs', inputs, 'labels', labels)
        len_x = arr_x.shape[0]
        batch_size = check_int_positive('batch_size', batch_size)
        batchs = int(len_x / batch_size)
        rest = len_x - batchs*batch_size
        res = []
        for i in range(batchs):
            x_batch = arr_x[i*batch_size: (i + 1)*batch_size]
            y_batch = arr_y[i*batch_size: (i + 1)*batch_size]
            adv_x = self.generate(x_batch, y_batch)
            # Black-attack methods will return 3 values, just get the second.
            res.append(adv_x[1] if isinstance(adv_x, tuple) else adv_x)

        if rest != 0:
            x_batch = arr_x[batchs*batch_size:]
            y_batch = arr_y[batchs*batch_size:]
            adv_x = self.generate(x_batch, y_batch)
            # Black-attack methods will return 3 values, just get the second.
            res.append(adv_x[1] if isinstance(adv_x, tuple) else adv_x)


        adv_x = np.concatenate(res, axis=0)
        return adv_x

    @abstractmethod
    def generate(self, inputs, labels):
        """
        Generate adversarial examples based on normal samples and their labels.

        Args:
            inputs (numpy.ndarray): Samples based on which adversarial
                examples are generated.
            labels (numpy.ndarray): Labels of samples, whose values determined
                by specific attacks.

        Raises:
            NotImplementedError: It is an abstract method.
        """
        msg = 'The function generate() is an abstract function in class ' \
              '`Attack` and should be implemented in child class.'
        LOGGER.error(TAG, msg)
        raise NotImplementedError(msg)

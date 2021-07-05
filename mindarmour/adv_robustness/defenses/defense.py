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
Base Class of Defense.
"""
from abc import abstractmethod

from mindarmour.utils.logger import LogUtil
from mindarmour.utils._check_param import check_pair_numpy_param, \
    check_int_positive

LOGGER = LogUtil.get_instance()
TAG = 'Defense'


class Defense:
    """
    The abstract base class for all defense classes defending adversarial
    examples.

    Args:
        network (Cell): A MindSpore-style deep learning model to be defensed.
    """

    def __init__(self, network):
        self._network = network

    @abstractmethod
    def defense(self, inputs, labels):
        """
        Defense model with samples.

        Args:
            inputs (numpy.ndarray): Samples based on which adversarial
                examples are generated.
            labels (numpy.ndarray): Labels of input samples.

        Raises:
            NotImplementedError: It is an abstract method.
        """
        msg = 'The function defense() is an abstract function in class ' \
              '`Defense` and should be implemented in child class.'
        LOGGER.error(TAG, msg)
        raise NotImplementedError(msg)

    def batch_defense(self, inputs, labels, batch_size=32, epochs=5):
        """
        Defense model with samples in batch.

        Args:
            inputs (numpy.ndarray): Samples based on which adversarial
                examples are generated.
            labels (numpy.ndarray): Labels of input samples.
            batch_size (int): Number of samples in one batch. Default: 32.
            epochs (int): Number of epochs. Default: 5.

        Returns:
            numpy.ndarray, loss of batch_defense operation.

        Raises:
            ValueError: If batch_size is 0.
        """
        inputs, labels = check_pair_numpy_param('inputs', inputs, 'labels',
                                                labels)
        x_len = len(inputs)
        batch_size = check_int_positive('batch_size', batch_size)

        iters_per_epoch = int(x_len / batch_size)
        loss = None
        for _ in range(epochs):
            for step in range(iters_per_epoch):
                x_batch = inputs[step*batch_size:(step + 1)*batch_size]
                y_batch = labels[step*batch_size:(step + 1)*batch_size]
                loss = self.defense(x_batch, y_batch)
        return loss

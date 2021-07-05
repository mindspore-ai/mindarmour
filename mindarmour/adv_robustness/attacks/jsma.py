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
JSMA-Attack.
"""
import numpy as np

from mindspore import Tensor
from mindspore.nn import Cell

from mindarmour.utils.util import GradWrap, jacobian_matrix
from mindarmour.utils.logger import LogUtil
from mindarmour.utils._check_param import check_pair_numpy_param, check_model, \
    check_param_type, check_int_positive, check_value_positive, \
    check_value_non_negative
from .attack import Attack

LOGGER = LogUtil.get_instance()
TAG = 'JSMA'


class JSMAAttack(Attack):
    """
    JSMA is an targeted & iterative attack based on saliency map of
    input features.

    Reference: `The limitations of deep learning in adversarial settings
    <https://arxiv.org/abs/1511.07528>`_

    Args:
        network (Cell): Target model.
        num_classes (int): Number of labels of model output, which should be
            greater than zero.
        box_min (float): Lower bound of input of the target model. Default: 0.
        box_max (float): Upper bound of input of the target model. Default: 1.0.
        theta (float): Change ratio of one pixel (relative to
               input data range). Default: 1.0.
        max_iteration (int): Maximum round of iteration. Default: 1000.
        max_count (int): Maximum times to change each pixel. Default: 3.
        increase (bool): If True, increase perturbation. If False, decrease
            perturbation. Default: True.
        sparse (bool): If True, input labels are sparse-coded. If False,
            input labels are onehot-coded. Default: True.

    Examples:
        >>> attack = JSMAAttack(network)
    """

    def __init__(self, network, num_classes, box_min=0.0, box_max=1.0,
                 theta=1.0, max_iteration=1000, max_count=3, increase=True,
                 sparse=True):
        super(JSMAAttack).__init__()
        LOGGER.debug(TAG, "init jsma class.")
        self._network = check_model('network', network, Cell)
        self._network.set_grad(True)
        self._min = check_value_non_negative('box_min', box_min)
        self._max = check_value_non_negative('box_max', box_max)
        self._num_classes = check_int_positive('num_classes', num_classes)
        self._theta = check_value_positive('theta', theta)
        self._max_iter = check_int_positive('max_iteration', max_iteration)
        self._max_count = check_int_positive('max_count', max_count)
        self._increase = check_param_type('increase', increase, bool)
        self._net_grad = GradWrap(self._network)
        self._bit_map = None
        self._sparse = check_param_type('sparse', sparse, bool)

    def _saliency_map(self, data, bit_map, target):
        """
        Compute the saliency map of all pixels.

        Args:
            data (numpy.ndarray): Input sample.
            bit_map (numpy.ndarray): Bit map to control modify frequency of
                each pixel.
            target (int): Target class.

        Returns:
            tuple, indices of selected pixel to modify.

        Examples:
            >>> p1_ind, p2_ind = self._saliency_map([0.2, 0.3, 0.5],
            >>>                                     [1, 0, 1], 1)
        """
        jaco_grad = jacobian_matrix(self._net_grad, data, self._num_classes)
        jaco_grad = jaco_grad.reshape(self._num_classes, -1)
        alpha = jaco_grad[target]*bit_map
        alpha_trans = np.reshape(alpha, (alpha.shape[0], 1))
        alpha_two_dim = alpha + alpha_trans
        # pixel influence on other classes except target class
        other_grads = [jaco_grad[class_ind] for class_ind in range(
            self._num_classes)]
        beta = np.sum(other_grads, axis=0)*bit_map - alpha
        beta_trans = np.reshape(beta, (beta.shape[0], 1))
        beta_two_dim = beta + beta_trans

        if self._increase:
            alpha_two_dim = (alpha_two_dim > 0)*alpha_two_dim
            beta_two_dim = (beta_two_dim < 0)*beta_two_dim
        else:
            alpha_two_dim = (alpha_two_dim < 0)*alpha_two_dim
            beta_two_dim = (beta_two_dim > 0)*beta_two_dim

        sal_map = (-1*alpha_two_dim*beta_two_dim)
        two_dim_index = np.argmax(sal_map)
        p1_ind = two_dim_index % len(data.flatten())
        p2_ind = two_dim_index // len(data.flatten())
        return p1_ind, p2_ind

    def _generate_one(self, data, target):
        """
        Generate one adversarial example.

        Args:
            data (numpy.ndarray): Input sample (only one).
            target (int): Target label.

        Returns:
            numpy.ndarray, adversarial example or zeros (if failed).

        Examples:
            >>> adv = self._generate_one([0.2, 0.3 ,0.4], 1)
        """
        ori_shape = data.shape
        temp = data.flatten()
        bit_map = np.ones_like(temp)
        fake_res = np.zeros_like(data)
        counter = np.zeros_like(temp)
        perturbed = np.copy(temp)
        for _ in range(self._max_iter):
            pre_logits = self._network(Tensor(np.expand_dims(
                perturbed.reshape(ori_shape), axis=0)))
            per_pred = np.argmax(pre_logits.asnumpy())
            if per_pred == target:
                LOGGER.debug(TAG, 'find one adversarial sample successfully.')
                return perturbed.reshape(ori_shape)
            if np.all(bit_map == 0):
                LOGGER.debug(TAG, 'fail to find adversarial sample')
                return perturbed.reshape(ori_shape)
            p1_ind, p2_ind = self._saliency_map(perturbed.reshape(
                ori_shape)[np.newaxis, :], bit_map, target)
            if self._increase:
                perturbed[p1_ind] += self._theta*(self._max - self._min)
                perturbed[p2_ind] += self._theta*(self._max - self._min)
            else:
                perturbed[p1_ind] -= self._theta*(self._max - self._min)
                perturbed[p2_ind] -= self._theta*(self._max - self._min)
            counter[p1_ind] += 1
            counter[p2_ind] += 1
            if (perturbed[p1_ind] >= self._max) or (
                    perturbed[p1_ind] <= self._min) \
                    or (counter[p1_ind] > self._max_count):
                bit_map[p1_ind] = 0
            if (perturbed[p2_ind] >= self._max) or (
                    perturbed[p2_ind] <= self._min) \
                    or (counter[p2_ind] > self._max_count):
                bit_map[p2_ind] = 0
            perturbed = np.clip(perturbed, self._min, self._max)
        LOGGER.debug(TAG, 'fail to find adversarial sample.')
        return fake_res

    def generate(self, inputs, labels):
        """
        Generate adversarial examples in batch.

        Args:
            inputs (numpy.ndarray): Input samples.
            labels (numpy.ndarray): Target labels.

        Returns:
            numpy.ndarray, adversarial samples.

        Examples:
            >>> advs = generate([[0.2, 0.3, 0.4], [0.3, 0.4, 0.5]], [1, 2])
        """
        inputs, labels = check_pair_numpy_param('inputs', inputs,
                                                'labels', labels)
        if not self._sparse:
            labels = np.argmax(labels, axis=1)
        LOGGER.debug(TAG, 'start to generate adversarial samples.')
        res = []
        for i in range(inputs.shape[0]):
            res.append(self._generate_one(inputs[i], labels[i]))
        LOGGER.debug(TAG, 'finished.')
        return np.asarray(res)

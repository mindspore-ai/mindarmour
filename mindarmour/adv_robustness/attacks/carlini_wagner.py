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
Carlini-wagner Attack.
"""
import numpy as np

from mindspore import Tensor
from mindspore.nn import Cell

from mindarmour.utils.logger import LogUtil
from mindarmour.utils._check_param import check_numpy_param, check_model, \
    check_pair_numpy_param, check_int_positive, check_param_type, \
    check_param_multi_types, check_value_positive, check_equal_shape
from mindarmour.utils.util import GradWrap, jacobian_matrix
from .attack import Attack

LOGGER = LogUtil.get_instance()
TAG = 'CW'


def _best_logits_of_other_class(logits, target_class, value=1):
    """
    Choose the index of the largest logits exclude target class.

    Args:
        logits (numpy.ndarray): Predict logits of samples.
        target_class (numpy.ndarray): Target labels.
        value (float): Maximum value of output logits. Default: 1.

    Returns:
        numpy.ndarray, the index of the largest logits exclude the target
        class.

    Examples:
        >>> other_class = _best_logits_of_other_class([[0.2, 0.3, 0.5],
        >>> [0.3, 0.4, 0.3]], [2, 1])
    """
    LOGGER.debug(TAG, "enter the func _best_logits_of_other_class.")
    logits, target_class = check_pair_numpy_param('logits', logits,
                                                  'target_class', target_class)
    res = np.zeros_like(logits)
    for i in range(logits.shape[0]):
        res[i][target_class[i]] = value
    return np.argmax(logits - res, axis=1)


class CarliniWagnerL2Attack(Attack):
    """
    The Carlini & Wagner attack using L2 norm.

    References: `Nicholas Carlini, David Wagner: "Towards Evaluating
    the Robustness of Neural Networks" <https://arxiv.org/abs/1608.04644>`_

    Args:
        network (Cell): Target model.
        num_classes (int): Number of labels of model output, which should be
            greater than zero.
        box_min (float): Lower bound of input of the target model. Default: 0.
        box_max (float): Upper bound of input of the target model. Default: 1.0.
        bin_search_steps (int): The number of steps for the binary search
            used to find the optimal trade-off constant between distance
            and confidence. Default: 5.
        max_iterations (int): The maximum number of iterations, which should be
            greater than zero. Default: 1000.
        confidence (float): Confidence of the output of adversarial examples.
            Default: 0.
        learning_rate (float): The learning rate for the attack algorithm.
            Default: 5e-3.
        initial_const (float): The initial trade-off constant to use to balance
            the relative importance of perturbation norm and confidence
            difference. Default: 1e-2.
        abort_early_check_ratio (float): Check loss progress every ratio of
            all iteration. Default: 5e-2.
        targeted (bool): If True, targeted attack. If False, untargeted attack.
            Default: False.
        fast (bool): If True, return the first found adversarial example.
            If False, return the adversarial samples with smaller
            perturbations. Default: True.
        abort_early (bool): If True, Adam will be aborted if the loss hasn't
            decreased for some time. If False, Adam will continue work until the
            max iterations is arrived. Default: True.
        sparse (bool): If True, input labels are sparse-coded. If False,
            input labels are onehot-coded. Default: True.

    Examples:
        >>> attack = CarliniWagnerL2Attack(network)
    """

    def __init__(self, network, num_classes, box_min=0.0, box_max=1.0,
                 bin_search_steps=5, max_iterations=1000, confidence=0,
                 learning_rate=5e-3, initial_const=1e-2,
                 abort_early_check_ratio=5e-2, targeted=False,
                 fast=True, abort_early=True, sparse=True):
        LOGGER.info(TAG, "init CW object.")
        super(CarliniWagnerL2Attack, self).__init__()
        self._network = check_model('network', network, Cell)
        self._network.set_grad(True)
        self._num_classes = check_int_positive('num_classes', num_classes)
        self._min = check_param_type('box_min', box_min, float)
        self._max = check_param_type('box_max', box_max, float)
        self._bin_search_steps = check_int_positive('search_steps',
                                                    bin_search_steps)
        self._max_iterations = check_int_positive('max_iterations',
                                                  max_iterations)
        self._confidence = check_param_multi_types('confidence', confidence,
                                                   [int, float])
        self._learning_rate = check_value_positive('learning_rate',
                                                   learning_rate)
        self._initial_const = check_value_positive('initial_const',
                                                   initial_const)
        self._abort_early = check_param_type('abort_early', abort_early, bool)
        self._fast = check_param_type('fast', fast, bool)
        self._abort_early_check_ratio = check_value_positive('abort_early_check_ratio',
                                                             abort_early_check_ratio)
        self._targeted = check_param_type('targeted', targeted, bool)
        self._net_grad = GradWrap(self._network)
        self._sparse = check_param_type('sparse', sparse, bool)
        self._dtype = None

    def _loss_function(self, logits, new_x, org_x, org_or_target_class,
                       constant, confidence):
        """
        Calculate the value of loss function and gradients of loss w.r.t inputs.

        Args:
            logits (numpy.ndarray): The output of network before softmax.
            new_x (numpy.ndarray): Adversarial examples.
            org_x (numpy.ndarray): Original benign input samples.
            org_or_target_class (numpy.ndarray): Original/target labels.
            constant (float): A trade-off constant to use to balance loss
                and perturbation norm.
            confidence (float): Confidence level of the output of adversarial
                examples.

        Returns:
            numpy.ndarray, norm of perturbation, sum of the loss and the
            norm, and gradients of the sum w.r.t inputs.

        Raises:
            ValueError: If loss is less than 0.

        Examples:
            >>> L2_loss, total_loss, dldx = self._loss_function([0.2 , 0.3,
            >>> 0.5], [0.1, 0.2, 0.2, 0.4], [0.12, 0.2, 0.25, 0.4], [1], 2, 0)
        """
        LOGGER.debug(TAG, "enter the func _loss_function.")

        logits = check_numpy_param('logits', logits)
        org_x = check_numpy_param('org_x', org_x)
        new_x, org_or_target_class = check_pair_numpy_param('new_x',
                                                            new_x,
                                                            'org_or_target_class',
                                                            org_or_target_class)

        new_x, org_x = check_equal_shape('new_x', new_x, 'org_x', org_x)

        other_class_index = _best_logits_of_other_class(
            logits, org_or_target_class, value=np.inf)
        loss1 = np.sum((new_x - org_x)**2,
                       axis=tuple(range(len(new_x.shape))[1:]))
        loss2 = np.zeros_like(loss1, dtype=self._dtype)
        loss2_grade = np.zeros_like(new_x, dtype=self._dtype)
        jaco_grad = jacobian_matrix(self._net_grad, new_x, self._num_classes)
        if self._targeted:
            for i in range(org_or_target_class.shape[0]):
                loss2[i] = max(0, logits[i][other_class_index[i]]
                               - logits[i][org_or_target_class[i]]
                               + confidence)
                loss2_grade[i] = constant[i]*(jaco_grad[other_class_index[
                    i]][i] - jaco_grad[org_or_target_class[i]][i])
        else:
            for i in range(org_or_target_class.shape[0]):
                loss2[i] = max(0, logits[i][org_or_target_class[i]]
                               - logits[i][other_class_index[i]] + confidence)
                loss2_grade[i] = constant[i]*(jaco_grad[org_or_target_class[
                    i]][i] - jaco_grad[other_class_index[i]][i])
        total_loss = loss1 + constant*loss2
        loss1_grade = 2*(new_x - org_x)
        for i in range(org_or_target_class.shape[0]):
            if loss2[i] < 0:
                msg = 'loss value should greater than or equal to 0, ' \
                      'but got loss2 {}'.format(loss2[i])
                LOGGER.error(TAG, msg)
                raise ValueError(msg)
            if loss2[i] == 0:
                loss2_grade[i, ...] = 0
        total_loss_grade = loss1_grade + loss2_grade
        return loss1, total_loss, total_loss_grade

    def _to_attack_space(self, inputs):
        """
        Transform input data into attack space.

        Args:
            inputs (numpy.ndarray): Input data.

        Returns:
            numpy.ndarray, transformed data which belongs to attack space.

        Examples:
            >>> x_att = self._to_attack_space([0.2, 0.3, 0.3])
        """
        LOGGER.debug(TAG, "enter the func _to_attack_space.")

        inputs = check_numpy_param('inputs', inputs)
        mean = (self._min + self._max) / 2
        diff = (self._max - self._min) / 2
        inputs = (inputs - mean) / diff
        inputs = inputs*0.999999
        return np.arctanh(inputs)

    def _to_model_space(self, inputs):
        """
        Transform input data into model space.

        Args:
            inputs (numpy.ndarray): Input data.

        Returns:
            numpy.ndarray, transformed data which belongs to model space
            and the gradient of x_model w.r.t. x_att.

        Examples:
            >>> x_att = self._to_model_space([10, 21, 9])
        """
        LOGGER.debug(TAG, "enter the func _to_model_space.")

        inputs = check_numpy_param('inputs', inputs)
        inputs = np.tanh(inputs)
        the_grad = 1 - np.square(inputs)
        mean = (self._min + self._max) / 2
        diff = (self._max - self._min) / 2
        inputs = inputs*diff + mean
        the_grad = the_grad*diff
        return inputs, the_grad

    def generate(self, inputs, labels):
        """
        Generate adversarial examples based on input data and targeted labels.

        Args:
            inputs (numpy.ndarray): Input samples.
            labels (numpy.ndarray): The ground truth label of input samples
                or target labels.

        Returns:
            numpy.ndarray, generated adversarial examples.

        Examples:
            >>> advs = attack.generate([[0.1, 0.2, 0.6], [0.3, 0, 0.4]], [1, 2]]
        """

        LOGGER.debug(TAG, "enter the func generate.")
        inputs, labels = check_pair_numpy_param('inputs', inputs,
                                                'labels', labels)
        if not self._sparse:
            labels = np.argmax(labels, axis=1)
        self._dtype = inputs.dtype
        att_original = self._to_attack_space(inputs)
        reconstructed_original, _ = self._to_model_space(att_original)

        # find an adversarial sample
        const = np.ones_like(labels, dtype=self._dtype)*self._initial_const
        lower_bound = np.zeros_like(labels, dtype=self._dtype)
        upper_bound = np.ones_like(labels, dtype=self._dtype)*np.inf
        adversarial_res = inputs.copy()
        adversarial_loss = np.ones_like(labels, dtype=self._dtype)*np.inf
        samples_num = labels.shape[0]
        adv_flag = np.zeros_like(labels)
        for binary_search_step in range(self._bin_search_steps):
            if (binary_search_step == self._bin_search_steps - 1) and \
                    (self._bin_search_steps >= 10):
                const = min(1e10, upper_bound)
            LOGGER.debug(TAG,
                         'starting optimization with const = %s',
                         str(const))

            att_perturbation = np.zeros_like(att_original, dtype=self._dtype)
            loss_at_previous_check = np.ones_like(labels, dtype=self._dtype)*np.inf

            # create a new optimizer to minimize the perturbation
            optimizer = _AdamOptimizer(att_perturbation.shape)

            for iteration in range(self._max_iterations):
                x_input, dxdp = self._to_model_space(
                    att_original + att_perturbation)
                logits = self._network(Tensor(x_input)).asnumpy()

                current_l2_loss, current_loss, dldx = self._loss_function(
                    logits, x_input, reconstructed_original,
                    labels, const, self._confidence)

                # check if attack success (include all examples)
                if self._targeted:
                    is_adv = (np.argmax(logits, axis=1) == labels)
                else:
                    is_adv = (np.argmax(logits, axis=1) != labels)

                for i in range(samples_num):
                    if is_adv[i]:
                        adv_flag[i] = True
                        if current_l2_loss[i] < adversarial_loss[i]:
                            adversarial_res[i] = x_input[i]
                            adversarial_loss[i] = current_l2_loss[i]

                if np.all(adv_flag):
                    if self._fast:
                        LOGGER.debug(TAG, "succeed find adversarial examples.")
                        msg = 'iteration: {}, logits_att: {}, ' \
                              'loss: {}, l2_dist: {}' \
                            .format(iteration,
                                    np.argmax(logits, axis=1),
                                    current_loss, current_l2_loss)
                        LOGGER.debug(TAG, msg)
                        return adversarial_res

                dldx, inputs = check_equal_shape('dldx', dldx, 'inputs', inputs)

                gradient = dldx*dxdp
                att_perturbation += \
                    optimizer(gradient, self._learning_rate)

                # check if should stop iteration early
                flag = True
                iter_check = iteration % (np.ceil(
                    self._max_iterations*self._abort_early_check_ratio))
                if self._abort_early and iter_check == 0:
                    # check progress
                    for i in range(inputs.shape[0]):
                        if current_loss[i] <= .9999*loss_at_previous_check[i]:
                            flag = False
                    # stop Adam if all samples has no progress
                    if flag:
                        LOGGER.debug(TAG,
                                     'step:%d, no progress yet, stop iteration',
                                     binary_search_step)
                        break
                    loss_at_previous_check = current_loss

            for i in range(samples_num):
                # update bound based on search result
                if adv_flag[i]:
                    LOGGER.debug(TAG,
                                 'example %d, found adversarial with const=%f',
                                 i, const[i])
                    upper_bound[i] = const[i]
                else:
                    LOGGER.debug(TAG,
                                 'example %d, failed to find adversarial'
                                 ' with const=%f',
                                 i, const[i])
                    lower_bound[i] = const[i]

                if upper_bound[i] == np.inf:
                    const[i] *= 10
                else:
                    const[i] = (lower_bound[i] + upper_bound[i]) / 2

        return adversarial_res


class _AdamOptimizer:
    """
    AdamOptimizer is used to calculate the optimum attack step.

    Args:
        shape (tuple): The shape of perturbations.

    Examples:
        >>> optimizer = _AdamOptimizer(att_perturbation.shape)
    """

    def __init__(self, shape):
        self._m = np.zeros(shape)
        self._v = np.zeros(shape)
        self._t = 0

    def __call__(self, gradient, learning_rate=0.001,
                 beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Calculate the optimum perturbation for each iteration.

        Args:
            gradient (numpy.ndarray): The gradient of the loss w.r.t. to the
                variable.
            learning_rate (float): The learning rate in the current iteration.
                Default: 0.001.
            beta1 (float): Decay rate for calculating the exponentially
                decaying average of past gradients. Default: 0.9.
            beta2 (float): Decay rate for calculating the exponentially
                decaying average of past squared gradients. Default: 0.999.
            epsilon (float): Small value to avoid division by zero.
                Default: 1e-8.

        Returns:
            numpy.ndarray, perturbations.

        Examples:
            >>> perturbs = optimizer([0.2, 0.1, 0.15], 0.005)
        """
        gradient = check_numpy_param('gradient', gradient)
        self._t += 1
        self._m = beta1*self._m + (1 - beta1)*gradient
        self._v = beta2*self._v + (1 - beta2)*gradient**2
        alpha = learning_rate*np.sqrt(1 - beta2**self._t) / (1 - beta1**self._t)
        pertur = -alpha*self._m / (np.sqrt(self._v) + epsilon)
        return pertur

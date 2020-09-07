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
LBFGS-Attack.
"""
import numpy as np
import scipy.optimize as so

from mindspore import Tensor
from mindspore.nn import Cell
from mindspore.nn import SoftmaxCrossEntropyWithLogits

from mindarmour.utils.logger import LogUtil
from mindarmour.utils.util import WithLossCell, GradWrapWithLoss
from mindarmour.utils._check_param import check_pair_numpy_param, check_model, \
    check_int_positive, check_value_positive, check_param_type, \
    check_param_multi_types
from .attack import Attack

LOGGER = LogUtil.get_instance()
TAG = 'LBFGS'


class LBFGS(Attack):
    """
    Uses L-BFGS-B to minimize the distance between the input and the adversarial example.

    References: `Pedro Tabacof, Eduardo Valle. "Exploring the Space of
    Adversarial Images" <https://arxiv.org/abs/1510.05328>`_

    Args:
        network (Cell): The network of attacked model.
        eps (float): Attack step size. Default: 1e-5.
        bounds (tuple): Upper and lower bounds of data. Default: (0.0, 1.0)
        is_targeted (bool): If True, targeted attack. If False, untargeted
            attack. Default: True.
        nb_iter (int): Number of iteration of lbfgs-optimizer, which should be
            greater than zero. Default: 150.
        search_iters (int): Number of changes in step size, which should be
            greater than zero. Default: 30.
        loss_fn (Functions): Loss function of substitute model. Default: None.
        sparse (bool): If True, input labels are sparse-coded. If False,
            input labels are onehot-coded. Default: False.

    Examples:
        >>> attack = LBFGS(network)
    """
    def __init__(self, network, eps=1e-5, bounds=(0.0, 1.0), is_targeted=True,
                 nb_iter=150, search_iters=30, loss_fn=None, sparse=False):
        super(LBFGS, self).__init__()
        self._network = check_model('network', network, Cell)
        self._eps = check_value_positive('eps', eps)
        self._is_targeted = check_param_type('is_targeted', is_targeted, bool)
        self._nb_iter = check_int_positive('nb_iter', nb_iter)
        self._search_iters = check_int_positive('search_iters', search_iters)
        if loss_fn is None:
            loss_fn = SoftmaxCrossEntropyWithLogits(sparse=False)
        with_loss_cell = WithLossCell(self._network, loss_fn)
        self._grad_all = GradWrapWithLoss(with_loss_cell)
        self._dtype = None
        self._bounds = check_param_multi_types('bounds', bounds, [list, tuple])
        self._sparse = check_param_type('sparse', sparse, bool)
        for b in self._bounds:
            _ = check_param_multi_types('bound', b, [int, float])
        box_max, box_min = bounds
        if box_max < box_min:
            self._box_min = box_max
            self._box_max = box_min
        else:
            self._box_min = box_min
            self._box_max = box_max

    def generate(self, inputs, labels):
        """
        Generate adversarial examples based on input data and target labels.

        Args:
            inputs (numpy.ndarray): Benign input samples used as references to create
                adversarial examples.
            labels (numpy.ndarray): Original/target labels.

        Returns:
            numpy.ndarray, generated adversarial examples.

        Examples:
            >>> adv = attack.generate([[0.1, 0.2, 0.6], [0.3, 0, 0.4]], [2, 2])
        """
        LOGGER.debug(TAG, 'start to generate adv image.')
        arr_x, arr_y = check_pair_numpy_param('inputs', inputs, 'labels', labels)
        self._dtype = arr_x.dtype
        adv_list = list()
        for original_x, label_y in zip(arr_x, arr_y):
            adv_list.append(self._optimize(
                original_x, label_y, epsilon=self._eps))
        return np.array(adv_list)

    def _forward_one(self, cur_input):
        """Forward one sample in model."""
        cur_input = np.expand_dims(cur_input, axis=0)
        out_logits = self._network(Tensor(cur_input)).asnumpy()
        return out_logits

    def _gradient(self, cur_input, labels, shape):
        """ Return model gradient to minimize loss in l-bfgs-b."""
        label_dtype = labels.dtype
        labels = np.expand_dims(labels, axis=0).astype(label_dtype)
        # input shape should like original shape
        reshape_input = np.expand_dims(cur_input.reshape(shape),
                                       axis=0)
        out_grad = self._grad_all(Tensor(reshape_input), Tensor(labels))
        if isinstance(out_grad, tuple):
            out_grad = out_grad[0]
        return out_grad.asnumpy()

    def _loss(self, cur_input, start_input, cur_eps, shape, labels):
        """
        The l-bfgs-b loss is the sum of l2 distances to the original input plus
        the cross-entropy loss.
        """
        cur_input = cur_input.astype(self._dtype)
        l2_distance = np.linalg.norm(
            cur_input.reshape((cur_input.shape[0], -1)) - start_input.reshape(
                (start_input.shape[0], -1)))
        logits = self._forward_one(cur_input.reshape(shape)).flatten()
        logits = logits - np.max(logits)
        if self._sparse:
            target_class = labels
        else:
            target_class = np.argmax(labels)
        if self._is_targeted:
            crossentropy = np.log(np.sum(np.exp(logits))) - logits[target_class]
            gradient = self._gradient(cur_input, labels, shape).flatten()
        else:
            crossentropy = logits[target_class] - np.log(np.sum(np.exp(logits)))
            gradient = -self._gradient(cur_input, labels, shape).flatten()

        return (l2_distance + cur_eps*crossentropy).astype(self._dtype), \
               gradient.astype(np.float64)

    def _lbfgsb(self, start_input, cur_eps, shape, labels, bounds):
        """
        A wrapper.
        Method reference to `scipy.optimize.fmin_l_bfgs_b`_

        .. _`scipy.optimize.fmin_l_bfgs_b`: https://docs.scipy.org/doc/scipy/
        reference/generated/scipy.optimize.fmin_l_bfgs_b.html
        """
        approx_grad_eps = (self._box_max - self._box_min) / 100
        max_matrix_variable = 15
        cur_input, _, detail_info = so.fmin_l_bfgs_b(
            self._loss,
            start_input,
            args=(start_input, cur_eps, shape, labels),
            approx_grad=False,
            bounds=bounds,
            m=max_matrix_variable,
            maxiter=self._nb_iter,
            epsilon=approx_grad_eps)

        LOGGER.debug(TAG, str(detail_info))
        # LBFGS-B does not always exactly respect the boundaries
        if np.amax(cur_input) > self._box_max or np.amin(
                cur_input) < self._box_min:  # pragma: no coverage
            LOGGER.debug(TAG,
                         'Input out of bounds (min, max = %s, %s).'
                         ' Performing manual clip.',
                         np.amin(cur_input),
                         np.amax(cur_input))
            cur_input = np.clip(cur_input, self._box_min, self._box_max)
        cur_input = cur_input.astype(self._dtype)
        cur_input = cur_input.reshape(shape)
        adv_prediction = self._forward_one(cur_input)

        LOGGER.debug(TAG, 'input one sample label is :{}'.format(labels))
        if not self._sparse:
            labels = np.argmax(labels)
        if self._is_targeted:
            return cur_input, np.argmax(adv_prediction) == labels
        return cur_input, np.argmax(adv_prediction) != labels

    def _optimize(self, start_input, labels, epsilon):
        """
        Given loss fuction and gradient, use l_bfgs_b algorithm to update input
        sample. The epsilon will be doubled until an adversarial example is found.

        Args:
            start_input (numpy.ndarray): Benign input samples used as references
                to create adversarial examples.
            labels (numpy.ndarray): Target labels.
            epsilon: (float): Attack step size.
            max_iter (int): Number of iteration.
        """
        # store the shape for later and operate on the flattened input
        ori_shape = start_input.shape
        start_input = start_input.flatten().astype(self._dtype)
        bounds = [self._bounds]*len(start_input)

        # finding initial cur_eps
        iter_c = epsilon
        for _ in range(self._search_iters):
            iter_c = 2*iter_c
            generate_x, is_adversarial = self._lbfgsb(start_input,
                                                      iter_c,
                                                      ori_shape,
                                                      labels,
                                                      bounds)
            LOGGER.debug(TAG, 'Tested iter_c = %f', iter_c)
            if is_adversarial:
                LOGGER.debug(TAG, 'find adversarial successfully.')
                return generate_x
            LOGGER.debug(TAG, 'failed to not adversarial.')
        return generate_x

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
DeepFool Attack.
"""
import numpy as np

from mindspore import Tensor
from mindspore.nn import Cell

from mindarmour.utils.logger import LogUtil
from mindarmour.utils.util import GradWrap, jacobian_matrix
from mindarmour.utils._check_param import check_pair_numpy_param, check_model, \
    check_value_positive, check_int_positive, check_norm_level, \
    check_param_multi_types, check_param_type
from .attack import Attack

LOGGER = LogUtil.get_instance()
TAG = 'DeepFool'


class DeepFool(Attack):
    """
    DeepFool is an untargeted & iterative attack achieved by moving the benign
    sample to the nearest classification boundary and crossing the boundary.

    Reference: `DeepFool: a simple and accurate method to fool deep neural
    networks <https://arxiv.org/abs/1511.04599>`_

    Args:
        network (Cell): Target model.
        num_classes (int): Number of labels of model output, which should be
            greater than zero.
        max_iters (int): Max iterations, which should be
            greater than zero. Default: 50.
        overshoot (float): Overshoot parameter. Default: 0.02.
        norm_level (int): Order of the vector norm. Possible values: np.inf
            or 2. Default: 2.
        bounds (tuple): Upper and lower bounds of data range. In form of (clip_min,
            clip_max). Default: None.
        sparse (bool): If True, input labels are sparse-coded. If False,
            input labels are onehot-coded. Default: True.

    Examples:
        >>> attack = DeepFool(network)
    """

    def __init__(self, network, num_classes, max_iters=50, overshoot=0.02,
                 norm_level=2, bounds=None, sparse=True):
        super(DeepFool, self).__init__()
        self._network = check_model('network', network, Cell)
        self._network.set_grad(True)
        self._max_iters = check_int_positive('max_iters', max_iters)
        self._overshoot = check_value_positive('overshoot', overshoot)
        self._norm_level = check_norm_level(norm_level)
        self._num_classes = check_int_positive('num_classes', num_classes)
        self._net_grad = GradWrap(self._network)
        self._bounds = check_param_multi_types('bounds', bounds, [list, tuple])
        self._sparse = check_param_type('sparse', sparse, bool)
        for b in self._bounds:
            _ = check_param_multi_types('bound', b, [int, float])

    def generate(self, inputs, labels):
        """
        Generate adversarial examples based on input samples and original labels.

        Args:
            inputs (numpy.ndarray): Input samples.
            labels (numpy.ndarray): Original labels.

        Returns:
            numpy.ndarray, adversarial examples.

        Raises:
            NotImplementedError: If norm_level is not in [2, np.inf, '2', 'inf'].

        Examples:
            >>> advs = generate([[0.2, 0.3, 0.4], [0.3, 0.4, 0.5]], [1, 2])
        """
        inputs, labels = check_pair_numpy_param('inputs', inputs,
                                                'labels', labels)
        if not self._sparse:
            labels = np.argmax(labels, axis=1)
        inputs_dtype = inputs.dtype
        iteration = 0
        origin_labels = labels
        cur_labels = origin_labels.copy()
        weight = np.squeeze(np.zeros(inputs.shape[1:]))
        r_tot = np.zeros(inputs.shape)
        x_origin = inputs
        while np.any(cur_labels == origin_labels) and iteration < self._max_iters:
            preds = self._network(Tensor(inputs)).asnumpy()
            grads = jacobian_matrix(self._net_grad, inputs, self._num_classes)
            for idx in range(inputs.shape[0]):
                diff_w = np.inf
                label = origin_labels[idx]
                if cur_labels[idx] != label:
                    continue
                for k in range(self._num_classes):
                    if k == label:
                        continue
                    w_k = grads[k, idx, ...] - grads[label, idx, ...]
                    f_k = preds[idx, k] - preds[idx, label]
                    if self._norm_level == 2 or self._norm_level == '2':
                        diff_w_k = abs(f_k) / (np.linalg.norm(w_k) + 1e-8)
                    elif self._norm_level == np.inf \
                            or self._norm_level == 'inf':
                        diff_w_k = abs(f_k) / (np.linalg.norm(w_k, ord=1) + 1e-8)
                    else:
                        msg = 'ord {} is not available.' \
                            .format(str(self._norm_level))
                        LOGGER.error(TAG, msg)
                        raise NotImplementedError(msg)
                    if diff_w_k < diff_w:
                        diff_w = diff_w_k
                        weight = w_k

                if self._norm_level == 2 or self._norm_level == '2':
                    r_i = diff_w*weight / (np.linalg.norm(weight) + 1e-8)
                elif self._norm_level == np.inf or self._norm_level == 'inf':
                    r_i = diff_w*np.sign(weight) \
                          / (np.linalg.norm(weight, ord=1) + 1e-8)
                else:
                    msg = 'ord {} is not available in normalization.' \
                        .format(str(self._norm_level))
                    LOGGER.error(TAG, msg)
                    raise NotImplementedError(msg)
                r_tot[idx, ...] = r_tot[idx, ...] + r_i

            if self._bounds is not None:
                clip_min, clip_max = self._bounds
                inputs = x_origin + (1 + self._overshoot)*r_tot*(clip_max
                                                                 - clip_min)
                inputs = np.clip(inputs, clip_min, clip_max)
            else:
                inputs = x_origin + (1 + self._overshoot)*r_tot
            cur_labels = np.argmax(
                self._network(Tensor(inputs.astype(inputs_dtype))).asnumpy(),
                axis=1)
            iteration += 1
            inputs = inputs.astype(inputs_dtype)
            del preds, grads
        return inputs

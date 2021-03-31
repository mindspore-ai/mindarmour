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
Attack evaluation.
"""

import numpy as np

from mindarmour.utils.logger import LogUtil
from mindarmour.utils._check_param import check_pair_numpy_param, \
    check_param_type, check_numpy_param
from mindarmour.utils.util import calculate_lp_distance, compute_ssim

LOGGER = LogUtil.get_instance()
TAG = 'AttackEvaluate'


class AttackEvaluate:
    """
    Evaluation metrics of attack methods.

    Args:
        inputs (numpy.ndarray): Original samples.
        labels (numpy.ndarray): Original samples' label by one-hot format.
        adv_inputs (numpy.ndarray): Adversarial samples generated from original
            samples.
        adv_preds (numpy.ndarray): Probability of all output classes of
            adversarial examples.
        targeted (bool): If True, it is a targeted attack. If False, it is an
            untargeted attack. Default: False.
        target_label (numpy.ndarray): Targeted classes of adversarial examples,
            which is one dimension whose size is adv_inputs.shape[0].
            Default: None.

    Raises:
        ValueError: If target_label is None when targeted is True.

    Examples:
        >>> x = np.random.normal(size=(3, 512, 512, 3))
        >>> adv_x = np.random.normal(size=(3, 512, 512, 3))
        >>> y = np.array([[0.1, 0.1, 0.2, 0.6],
        >>>               [0.1, 0.7, 0.0, 0.2],
        >>>               [0.8, 0.1, 0.0, 0.1]])
        >>> adv_y = np.array([[0.1, 0.1, 0.2, 0.6],
        >>>                   [0.1, 0.0, 0.8, 0.1],
        >>>                   [0.0, 0.9, 0.1, 0.0]])
        >>> attack_eval = AttackEvaluate(x, y, adv_x, adv_y)
        >>> mr = attack_eval.mis_classification_rate()
    """

    def __init__(self, inputs, labels, adv_inputs, adv_preds,
                 targeted=False, target_label=None):
        self._inputs, self._labels = check_pair_numpy_param('inputs',
                                                            inputs,
                                                            'labels',
                                                            labels)
        self._adv_inputs, self._adv_preds = check_pair_numpy_param('adv_inputs',
                                                                   adv_inputs,
                                                                   'adv_preds',
                                                                   adv_preds)
        targeted = check_param_type('targeted', targeted, bool)
        self._targeted = targeted
        if target_label is not None:
            target_label = check_numpy_param('target_label', target_label)
        self._target_label = target_label
        self._true_label = np.argmax(self._labels, axis=1)
        self._adv_label = np.argmax(self._adv_preds, axis=1)

        idxes = np.arange(self._adv_preds.shape[0])
        if self._targeted:
            if target_label is None:
                msg = 'targeted attack need target_label, but got None.'
                LOGGER.error(TAG, msg)
                raise ValueError(msg)
            self._adv_preds, self._target_label = check_pair_numpy_param('adv_pred',
                                                                         self._adv_preds,
                                                                         'target_label',
                                                                         target_label)
            self._success_idxes = idxes[self._adv_label == self._target_label]
        else:
            self._success_idxes = idxes[self._adv_label != self._true_label]

    def mis_classification_rate(self):
        """
        Calculate misclassification rate(MR).

        Returns:
            float, ranges between (0, 1). The higher, the more successful the attack is.
        """
        return self._success_idxes.shape[0]*1.0 / self._inputs.shape[0]

    def avg_conf_adv_class(self):
        """
        Calculate average confidence of adversarial class (ACAC).

        Returns:
            float, ranges between (0, 1). The higher, the more successful the attack is.
        """
        idxes = self._success_idxes
        success_num = idxes.shape[0]
        if success_num == 0:
            return 0
        if self._targeted:
            return np.mean(self._adv_preds[idxes, self._target_label[idxes]])
        return np.mean(self._adv_preds[idxes, self._adv_label[idxes]])

    def avg_conf_true_class(self):
        """
        Calculate average confidence of true class (ACTC).

        Returns:
            float, ranges between (0, 1). The lower, the more successful the attack is.
        """
        idxes = self._success_idxes
        success_num = idxes.shape[0]
        if success_num == 0:
            return 0
        return np.mean(self._adv_preds[idxes, self._true_label[idxes]])

    def avg_lp_distance(self):
        """
        Calculate average lp distance (lp-dist).

        Returns:
            - float, return average l0, l2, or linf distance of all success
              adversarial examples, return value includes following cases.

              - If return value :math:`>=` 0, average lp distance. The lower,
                the more successful the attack is.

              - If return value is -1, there is no success adversarial examples.
        """
        idxes = self._success_idxes
        success_num = idxes.shape[0]
        if success_num == 0:
            return -1, -1, -1
        l0_dist = 0
        l2_dist = 0
        linf_dist = 0
        for i in idxes:
            l0_dist_i, l2_dist_i, linf_dist_i = calculate_lp_distance(self._inputs[i], self._adv_inputs[i])
            l0_dist += l0_dist_i
            l2_dist += l2_dist_i
            linf_dist += linf_dist_i

        return l0_dist / success_num, l2_dist / success_num, \
               linf_dist / success_num

    def avg_ssim(self):
        """
        Calculate average structural similarity (ASS).

        Returns:
            - float, average structural similarity.

              - If return value ranges between (0, 1), the higher, the more
                successful the attack is.

              - If return value is -1: there is no success adversarial examples.
        """
        success_num = self._success_idxes.shape[0]
        if success_num == 0:
            return -1

        total_ssim = 0.0
        for _, i in enumerate(self._success_idxes):
            total_ssim += compute_ssim(self._adv_inputs[i], self._inputs[i])

        return total_ssim / success_num

    def nte(self):
        """
        Calculate noise tolerance estimation (NTE).

        References: `Towards Imperceptible and Robust Adversarial Example Attacks
        against Neural Networks <https://arxiv.org/abs/1801.04693>`_


        Returns:
            float, ranges between (0, 1). The higher, the more successful the
            attack is.
        """
        idxes = self._success_idxes
        success_num = idxes.shape[0]
        adv_y = self._adv_preds[idxes]
        adv_y_2 = np.copy(adv_y)
        adv_y_2[range(success_num), np.argmax(adv_y_2, axis=1)] = 0
        net = np.mean(np.abs(np.max(adv_y_2, axis=1) - np.max(adv_y, axis=1)))

        return net

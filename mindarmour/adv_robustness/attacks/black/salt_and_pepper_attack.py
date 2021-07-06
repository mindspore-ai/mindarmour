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
SaltAndPepperNoise-Attack.
"""
import time
import numpy as np

from mindarmour.utils._check_param import check_model, check_pair_numpy_param, \
    check_param_type, check_int_positive, check_param_multi_types
from mindarmour.utils.logger import LogUtil
from ..attack import Attack
from .black_model import BlackModel

LOGGER = LogUtil.get_instance()
TAG = 'SaltAndPepperNoise-Attack'


class SaltAndPepperNoiseAttack(Attack):
    """
    Increases the amount of salt and pepper noise  to generate adversarial samples.

    Args:
        model (BlackModel): Target model.
        bounds (tuple): Upper and lower bounds of data. In form of (clip_min, clip_max). Default: (0.0, 1.0)
        max_iter (int): Max iteration to generate an adversarial example. Default: 100
        is_targeted (bool): If True, targeted attack. If False, untargeted attack. Default: False.
        sparse (bool): If True, input labels are sparse-encoded. If False, input labels are one-hot-encoded.
            Default: True.

    Examples:
        >>> attack = SaltAndPepperNoiseAttack(model)
    """

    def __init__(self, model, bounds=(0.0, 1.0), max_iter=100, is_targeted=False, sparse=True):
        super(SaltAndPepperNoiseAttack, self).__init__()
        self._model = check_model('model', model, BlackModel)
        self._bounds = check_param_multi_types('bounds', bounds, [tuple, list])
        for b in self._bounds:
            _ = check_param_multi_types('bound', b, [int, float])
        self._max_iter = check_int_positive('max_iter', max_iter)
        self._is_targeted = check_param_type('is_targeted', is_targeted, bool)
        self._sparse = check_param_type('sparse', sparse, bool)

    def generate(self, inputs, labels):
        """
        Generate adversarial examples based on input data and target labels.

        Args:
            inputs (numpy.ndarray): The original, unperturbed inputs.
            labels (numpy.ndarray): The target labels.

        Returns:
            - numpy.ndarray, bool values for each attack result.

            - numpy.ndarray, generated adversarial examples.

            - numpy.ndarray, query times for each sample.

        Examples:
            >>> adv_list = attack.generate(([[0.1, 0.2, 0.6], [0.3, 0, 0.4]], [1, 2])
        """
        arr_x, arr_y = check_pair_numpy_param('inputs', inputs, 'labels', labels)
        if not self._sparse:
            arr_y = np.argmax(arr_y, axis=1)

        is_adv_list = list()
        adv_list = list()
        query_times_each_adv = list()
        for sample, label in zip(arr_x, arr_y):
            start_t = time.time()
            is_adv, perturbed, query_times = self._generate_one(sample, label)
            is_adv_list.append(is_adv)
            adv_list.append(perturbed)
            query_times_each_adv.append(query_times)
            LOGGER.info(TAG, 'Finished one sample, adversarial is {}, cost time {:.2}s'.format(is_adv,
                                                                                               time.time() - start_t))
        is_adv_list = np.array(is_adv_list)
        adv_list = np.array(adv_list)
        query_times_each_adv = np.array(query_times_each_adv)
        return is_adv_list, adv_list, query_times_each_adv

    def _generate_one(self, one_input, label, epsilons=10):
        """
        Increases the amount of salt and pepper noise to generate adversarial samples.

        Args:
            one_input (numpy.ndarray): The original, unperturbed input.
            label (numpy.ndarray): The target label.
            epsilons (int) : Number of steps to try probability between 0 and 1. Default: 10

        Returns:
            - numpy.ndarray, bool values for result.

            - numpy.ndarray, adversarial example.

            - numpy.ndarray, query times for this sample.

        Examples:
            >>> one_adv = self._generate_one(input, label)
        """
        # use binary search to get epsilons
        low_ = 0.0
        high_ = 1.0
        query_count = 0
        input_shape = one_input.shape
        one_input = one_input.reshape(-1)
        best_adv = np.copy(one_input)
        best_eps = high_
        find_adv = False
        for _ in range(self._max_iter):
            min_eps = low_
            max_eps = (low_ + high_) / 2
            for _ in range(epsilons):
                adv = np.copy(one_input)
                noise = np.random.uniform(low=low_, high=high_, size=one_input.size)
                eps = (min_eps + max_eps) / 2
                # add salt
                adv[noise < eps] = self._bounds[0]
                # add pepper
                adv[noise >= (high_ - eps)] = self._bounds[1]
                query_count += 1
                ite_bool = self._model.is_adversarial(adv.reshape(input_shape), label, is_targeted=self._is_targeted)
                if ite_bool:
                    find_adv = True
                    if best_eps > eps:
                        best_adv = adv
                        best_eps = eps
                    max_eps = eps
                    LOGGER.debug(TAG, 'Attack succeed, epsilon is {}'.format(eps))
                else:
                    min_eps = eps
            if find_adv:
                break
        return find_adv, best_adv.reshape(input_shape), query_count

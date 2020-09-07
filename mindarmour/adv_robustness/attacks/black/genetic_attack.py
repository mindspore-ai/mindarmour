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
Genetic-Attack.
"""
import numpy as np
from scipy.special import softmax

from mindarmour.utils.logger import LogUtil
from mindarmour.utils._check_param import check_numpy_param, check_model, \
    check_pair_numpy_param, check_param_type, check_value_positive, \
    check_int_positive, check_param_multi_types
from ..attack import Attack
from .black_model import BlackModel

LOGGER = LogUtil.get_instance()
TAG = 'GeneticAttack'


def _mutation(cur_pop, step_noise=0.01, prob=0.005):
    """
    Generate mutation samples in genetic_attack.

    Args:
        cur_pop (numpy.ndarray): Samples before mutation.
        step_noise (float): Noise range. Default: 0.01.
        prob (float): Mutation probability. Default: 0.005.

    Returns:
        numpy.ndarray, samples after mutation operation in genetic_attack.

    Examples:
        >>> mul_pop = self._mutation_op([0.2, 0.3, 0.4], step_noise=0.03,
        >>> prob=0.01)
    """
    cur_pop = check_numpy_param('cur_pop', cur_pop)
    perturb_noise = np.clip(np.random.random(cur_pop.shape) - 0.5,
                            -step_noise, step_noise)
    mutated_pop = perturb_noise*(
        np.random.random(cur_pop.shape) < prob) + cur_pop
    return mutated_pop


class GeneticAttack(Attack):
    """
    The Genetic Attack represents the black-box attack based on the genetic algorithm,
    which belongs to differential evolution algorithms.

    This attack was proposed by Moustafa Alzantot et al. (2018).

    References: `Moustafa Alzantot, Yash Sharma, Supriyo Chakraborty,
    "GeneticAttack: Practical Black-box Attacks with
    Gradient-FreeOptimization" <https://arxiv.org/abs/1805.11090>`_

    Args:
        model (BlackModel): Target model.
        pop_size (int): The number of particles, which should be greater than
            zero. Default: 6.
        mutation_rate (float): The probability of mutations. Default: 0.005.
        per_bounds (float): Maximum L_inf distance.
        max_steps (int): The maximum round of iteration for each adversarial
            example. Default: 1000.
        step_size (float): Attack step size. Default: 0.2.
        temp (float): Sampling temperature for selection. Default: 0.3.
        bounds (tuple): Upper and lower bounds of data. In form of (clip_min,
            clip_max). Default: (0, 1.0)
        adaptive (bool): If True, turns on dynamic scaling of mutation
            parameters. If false, turns on static mutation parameters.
            Default: False.
        sparse (bool): If True, input labels are sparse-encoded. If False,
            input labels are one-hot-encoded. Default: True.

    Examples:
        >>> attack = GeneticAttack(model)
    """
    def __init__(self, model, pop_size=6,
                 mutation_rate=0.005, per_bounds=0.15, max_steps=1000,
                 step_size=0.20, temp=0.3, bounds=(0, 1.0), adaptive=False,
                 sparse=True):
        super(GeneticAttack, self).__init__()
        self._model = check_model('model', model, BlackModel)
        self._per_bounds = check_value_positive('per_bounds', per_bounds)
        self._pop_size = check_int_positive('pop_size', pop_size)
        self._step_size = check_value_positive('step_size', step_size)
        self._temp = check_value_positive('temp', temp)
        self._max_steps = check_int_positive('max_steps', max_steps)
        self._mutation_rate = check_value_positive('mutation_rate',
                                                   mutation_rate)
        self._adaptive = check_param_type('adaptive', adaptive, bool)
        self._bounds = check_param_multi_types('bounds', bounds, [list, tuple])
        for b in self._bounds:
            _ = check_param_multi_types('bound', b, [int, float])
        # initial global optimum fitness value
        self._best_fit = -1
        # count times of no progress
        self._plateau_times = 0
        # count times of changing attack step
        self._adap_times = 0
        self._sparse = check_param_type('sparse', sparse, bool)

    def generate(self, inputs, labels):
        """
        Generate adversarial examples based on input data and targeted
        labels (or ground_truth labels).

        Args:
            inputs (numpy.ndarray): Input samples.
            labels (numpy.ndarray): Targeted labels.

        Returns:
            - numpy.ndarray, bool values for each attack result.

            - numpy.ndarray, generated adversarial examples.

            - numpy.ndarray, query times for each sample.

        Examples:
            >>> advs = attack.generate([[0.2, 0.3, 0.4],
            >>>                         [0.3, 0.3, 0.2]],
            >>>                        [1, 2])
        """
        inputs, labels = check_pair_numpy_param('inputs', inputs,
                                                'labels', labels)
        # if input is one-hot encoded, get sparse format value
        if not self._sparse:
            if labels.ndim != 2:
                raise ValueError('labels must be 2 dims, '
                                 'but got {} dims.'.format(labels.ndim))
            labels = np.argmax(labels, axis=1)
        adv_list = []
        success_list = []
        query_times_list = []
        for i in range(inputs.shape[0]):
            is_success = False
            target_label = labels[i]
            iters = 0
            x_ori = inputs[i]
            # generate particles
            ori_copies = np.repeat(
                x_ori[np.newaxis, :], self._pop_size, axis=0)
            # initial perturbations
            cur_pert = np.clip(np.random.random(ori_copies.shape)*self._step_size,
                               (0 - self._per_bounds),
                               self._per_bounds)
            query_times = 0
            while iters < self._max_steps:
                iters += 1
                cur_pop = np.clip(
                    ori_copies + cur_pert, self._bounds[0], self._bounds[1])
                pop_preds = self._model.predict(cur_pop)
                query_times += cur_pop.shape[0]
                all_preds = np.argmax(pop_preds, axis=1)
                success_pop = np.equal(target_label, all_preds).astype(np.int32)
                success = max(success_pop)
                if success == 1:
                    is_success = True
                    adv = cur_pop[np.argmax(success_pop)]
                    break
                target_preds = pop_preds[:, target_label]
                others_preds_sum = np.sum(pop_preds, axis=1) - target_preds
                fit_vals = target_preds - others_preds_sum
                best_fit = max(target_preds - np.max(pop_preds))
                if best_fit > self._best_fit:
                    self._best_fit = best_fit
                    self._plateau_times = 0
                else:
                    self._plateau_times += 1
                adap_threshold = (lambda z: 100 if z > -0.4 else 300)(best_fit)
                if self._plateau_times > adap_threshold:
                    self._adap_times += 1
                    self._plateau_times = 0
                if self._adaptive:
                    step_noise = max(self._step_size, 0.4*(0.9**self._adap_times))
                    step_p = max(self._step_size, 0.5*(0.9**self._adap_times))
                else:
                    step_noise = self._step_size
                    step_p = self._mutation_rate
                step_temp = self._temp
                elite = cur_pert[np.argmax(fit_vals)]
                select_probs = softmax(fit_vals/step_temp)
                select_args = np.arange(self._pop_size)
                parents_arg = np.random.choice(
                    a=select_args, size=2*(self._pop_size - 1),
                    replace=True, p=select_probs)
                parent1 = cur_pert[parents_arg[:self._pop_size - 1]]
                parent2 = cur_pert[parents_arg[self._pop_size - 1:]]
                parent1_probs = select_probs[parents_arg[:self._pop_size - 1]]
                parent2_probs = select_probs[parents_arg[self._pop_size - 1:]]
                parent2_probs = parent2_probs / (parent1_probs + parent2_probs)
                # duplicate the probabilities to all features of each particle.
                dims = len(x_ori.shape)
                for _ in range(dims):
                    parent2_probs = parent2_probs[:, np.newaxis]
                parent2_probs = np.tile(parent2_probs, ((1,) + x_ori.shape))
                cross_probs = (np.random.random(parent1.shape) >
                               parent2_probs).astype(np.int32)
                childs = parent1*cross_probs + parent2*(1 - cross_probs)
                mutated_childs = _mutation(
                    childs, step_noise=self._per_bounds*step_noise,
                    prob=step_p)
                cur_pert = np.concatenate((mutated_childs, elite[np.newaxis, :]))
            if is_success:
                LOGGER.debug(TAG, 'successfully find one adversarial sample '
                                  'and start Reduction process.')
                adv_list.append(adv)
            else:
                LOGGER.debug(TAG, 'fail to find adversarial sample.')
                adv_list.append(elite + x_ori)
            LOGGER.debug(TAG,
                         'iteration times is: %d and query times is: %d',
                         iters,
                         query_times)
            success_list.append(is_success)
            query_times_list.append(query_times)
            del ori_copies, cur_pert, cur_pop
        return np.asarray(success_list), \
               np.asarray(adv_list), \
               np.asarray(query_times_list)

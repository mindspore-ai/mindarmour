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
PSO-Attack.
"""
import numpy as np

from mindarmour.utils.logger import LogUtil
from mindarmour.utils._check_param import check_model, check_pair_numpy_param, \
    check_numpy_param, check_value_positive, check_int_positive, \
    check_param_type, check_equal_shape, check_param_multi_types
from ..attack import Attack
from .black_model import BlackModel

LOGGER = LogUtil.get_instance()
TAG = 'PSOAttack'


class PSOAttack(Attack):
    """
    The PSO Attack represents the black-box attack based on Particle Swarm
    Optimization algorithm, which belongs to differential evolution algorithms.
    This attack was proposed by Rayan Mosli et al. (2019).

    References: `Rayan Mosli, Matthew Wright, Bo Yuan, Yin Pan, "They Might NOT
    Be Giants: Crafting Black-Box Adversarial Examples with Fewer Queries
    Using Particle Swarm Optimization", arxiv: 1909.07490, 2019.
    <https://arxiv.org/abs/1909.07490>`_

    Args:
        model (BlackModel): Target model.
        step_size (float): Attack step size. Default: 0.5.
        per_bounds (float): Relative variation range of perturbations. Default: 0.6.
        c1 (float): Weight coefficient. Default: 2.
        c2 (float): Weight coefficient. Default: 2.
        c (float): Weight of perturbation loss. Default: 2.
        pop_size (int): The number of particles, which should be greater
            than zero. Default: 6.
        t_max (int): The maximum round of iteration for each adversarial example,
            which should be greater than zero. Default: 1000.
        pm (float): The probability of mutations. Default: 0.5.
        bounds (tuple): Upper and lower bounds of data. In form of (clip_min,
            clip_max). Default: None.
        targeted (bool): If True, turns on the targeted attack. If False,
            turns on untargeted attack. Default: False.
        reduction_iters (int): Cycle times in reduction process. Default: 3.
        sparse (bool): If True, input labels are sparse-encoded. If False,
            input labels are one-hot-encoded. Default: True.

    Examples:
        >>> attack = PSOAttack(model)
    """

    def __init__(self, model, step_size=0.5, per_bounds=0.6, c1=2.0, c2=2.0,
                 c=2.0, pop_size=6, t_max=1000, pm=0.5, bounds=None,
                 targeted=False, reduction_iters=3, sparse=True):
        super(PSOAttack, self).__init__()
        self._model = check_model('model', model, BlackModel)
        self._step_size = check_value_positive('step_size', step_size)
        self._per_bounds = check_value_positive('per_bounds', per_bounds)
        self._c1 = check_value_positive('c1', c1)
        self._c2 = check_value_positive('c2', c2)
        self._c = check_value_positive('c', c)
        self._pop_size = check_int_positive('pop_size', pop_size)
        self._pm = check_value_positive('pm', pm)
        self._bounds = check_param_multi_types('bounds', bounds, [list, tuple])
        for b in self._bounds:
            _ = check_param_multi_types('bound', b, [int, float])
        self._targeted = check_param_type('targeted', targeted, bool)
        self._t_max = check_int_positive('t_max', t_max)
        self._reduce_iters = check_int_positive('reduction_iters',
                                                reduction_iters)
        self._sparse = check_param_type('sparse', sparse, bool)

    def _fitness(self, confi_ori, confi_adv, x_ori, x_adv):
        """
        Calculate the fitness value for each particle.

        Args:
            confi_ori (float): Maximum confidence or target label confidence of
                the original benign inputs' prediction confidences.
            confi_adv (float): Maximum confidence or target label confidence of
                the adversarial samples' prediction confidences.
            x_ori (numpy.ndarray): Benign samples.
            x_adv (numpy.ndarray): Adversarial samples.

        Returns:
            - float, fitness values of adversarial particles.

            - int, query times after reduction.

        Examples:
            >>> fitness = self._fitness(2.4, 1.2, [0.2, 0.3, 0.1], [0.21,
            >>> 0.34, 0.13])
        """
        x_ori = check_numpy_param('x_ori', x_ori)
        x_adv = check_numpy_param('x_adv', x_adv)
        fit_value = abs(
            confi_ori - confi_adv) - self._c / self._pop_size*np.linalg.norm(
                (x_adv - x_ori).reshape(x_adv.shape[0], -1), axis=1)
        return fit_value

    def _mutation_op(self, cur_pop):
        """
        Generate mutation samples.
        """
        cur_pop = check_numpy_param('cur_pop', cur_pop)
        perturb_noise = np.random.random(cur_pop.shape) - 0.5
        mutated_pop = perturb_noise*(np.random.random(cur_pop.shape)
                                     < self._pm) + cur_pop
        mutated_pop = np.clip(mutated_pop, cur_pop*(1 - self._per_bounds),
                              cur_pop*(1 + self._per_bounds))
        return mutated_pop

    def _reduction(self, x_ori, q_times, label, best_position):
        """
        Decrease the differences between the original samples and adversarial samples.

        Args:
            x_ori (numpy.ndarray): Original samples.
            q_times (int): Query times.
            label (int): Target label ot ground-truth label.
            best_position (numpy.ndarray): Adversarial examples.

        Returns:
            numpy.ndarray, adversarial examples after reduction.

        Examples:
            >>> adv_reduction = self._reduction(self, [0.1, 0.2, 0.3], 20, 1,
            >>> [0.12, 0.15, 0.25])
        """
        x_ori = check_numpy_param('x_ori', x_ori)
        best_position = check_numpy_param('best_position', best_position)
        x_ori, best_position = check_equal_shape('x_ori', x_ori,
                                                 'best_position', best_position)
        x_ori_fla = x_ori.flatten()
        best_position_fla = best_position.flatten()
        pixel_deep = self._bounds[1] - self._bounds[0]
        nums_pixel = len(x_ori_fla)
        for i in range(nums_pixel):
            diff = x_ori_fla[i] - best_position_fla[i]
            if abs(diff) > pixel_deep*0.1:
                old_poi_fla = np.copy(best_position_fla)
                best_position_fla[i] = np.clip(
                    best_position_fla[i] + diff*0.5,
                    self._bounds[0], self._bounds[1])
                cur_label = np.argmax(
                    self._model.predict(np.expand_dims(
                        best_position_fla.reshape(x_ori.shape), axis=0))[0])
                q_times += 1
                if self._targeted:
                    if cur_label != label:
                        best_position_fla = old_poi_fla
                else:
                    if cur_label == label:
                        best_position_fla = old_poi_fla
        return best_position_fla.reshape(x_ori.shape), q_times

    def generate(self, inputs, labels):
        """
        Generate adversarial examples based on input data and targeted
        labels (or ground_truth labels).

        Args:
            inputs (numpy.ndarray): Input samples.
            labels (numpy.ndarray): Targeted labels or ground_truth labels.

        Returns:
            - numpy.ndarray, bool values for each attack result.

            - numpy.ndarray, generated adversarial examples.

            - numpy.ndarray, query times for each sample.

        Examples:
            >>> advs = attack.generate([[0.2, 0.3, 0.4], [0.3, 0.3, 0.2]],
            >>> [1, 2])
        """
        inputs, labels = check_pair_numpy_param('inputs', inputs,
                                                'labels', labels)
        if not self._sparse:
            labels = np.argmax(labels, axis=1)
        # generate one adversarial each time
        if self._targeted:
            target_labels = labels
        adv_list = []
        success_list = []
        query_times_list = []
        pixel_deep = self._bounds[1] - self._bounds[0]
        for i in range(inputs.shape[0]):
            is_success = False
            q_times = 0
            x_ori = inputs[i]
            confidences = self._model.predict(np.expand_dims(x_ori, axis=0))[0]
            q_times += 1
            true_label = labels[i]
            if self._targeted:
                t_label = target_labels[i]
                confi_ori = confidences[t_label]
            else:
                confi_ori = max(confidences)
            # step1, initializing
            # initial global optimum fitness value, cannot set to be 0
            best_fitness = -np.inf
            # initial global optimum position
            best_position = x_ori
            x_copies = np.repeat(x_ori[np.newaxis, :], self._pop_size, axis=0)
            cur_noise = np.clip((np.random.random(x_copies.shape) - 0.5)
                                *self._step_size,
                                (0 - self._per_bounds)*(x_copies + 0.1),
                                self._per_bounds*(x_copies + 0.1))
            par = np.clip(x_copies + cur_noise,
                          x_copies*(1 - self._per_bounds),
                          x_copies*(1 + self._per_bounds))
            # initial advs
            par_ori = np.copy(par)
            # initial optimum positions for particles
            par_best_poi = np.copy(par)
            # initial optimum fitness values
            par_best_fit = -np.inf*np.ones(self._pop_size)
            # step2, optimization
            # initial velocities for particles
            v_particles = np.zeros(par.shape)
            is_mutation = False
            iters = 0
            while iters < self._t_max:
                last_best_fit = best_fitness
                ran_1 = np.random.random(par.shape)
                ran_2 = np.random.random(par.shape)
                v_particles = self._step_size*(
                    v_particles + self._c1*ran_1*(best_position - par)) \
                              + self._c2*ran_2*(par_best_poi - par)
                par = np.clip(par + v_particles,
                              (par_ori + 0.1*pixel_deep)*(
                                  1 - self._per_bounds),
                              (par_ori + 0.1*pixel_deep)*(
                                  1 + self._per_bounds))
                if iters > 30 and is_mutation:
                    par = self._mutation_op(par)
                if self._targeted:
                    confi_adv = self._model.predict(par)[:, t_label]
                else:
                    confi_adv = np.max(self._model.predict(par), axis=1)
                q_times += self._pop_size
                fit_value = self._fitness(confi_ori, confi_adv, x_ori, par)
                for k in range(self._pop_size):
                    if fit_value[k] > par_best_fit[k]:
                        par_best_fit[k] = fit_value[k]
                        par_best_poi[k] = par[k]
                    if fit_value[k] > best_fitness:
                        best_fitness = fit_value[k]
                        best_position = par[k]
                iters += 1
                cur_pre = self._model.predict(np.expand_dims(best_position,
                                                             axis=0))[0]
                is_mutation = False
                if (best_fitness - last_best_fit) < last_best_fit*0.05:
                    is_mutation = True
                cur_label = np.argmax(cur_pre)
                q_times += 1
                if self._targeted:
                    if cur_label == t_label:
                        is_success = True
                else:
                    if cur_label != true_label:
                        is_success = True
                if is_success:
                    LOGGER.debug(TAG, 'successfully find one adversarial '
                                      'sample and start Reduction process')
                    # step3, reduction
                    if self._targeted:
                        best_position, q_times = self._reduction(
                            x_ori, q_times, t_label, best_position)
                    else:
                        best_position, q_times = self._reduction(
                            x_ori, q_times, true_label, best_position)
                    break
            if not is_success:
                LOGGER.debug(TAG,
                             'fail to find adversarial sample, iteration '
                             'times is: %d and query times is: %d',
                             iters,
                             q_times)
            adv_list.append(best_position)
            success_list.append(is_success)
            query_times_list.append(q_times)
            del x_copies, cur_noise, par, par_ori, par_best_poi
        return np.asarray(success_list), \
               np.asarray(adv_list), \
               np.asarray(query_times_list)

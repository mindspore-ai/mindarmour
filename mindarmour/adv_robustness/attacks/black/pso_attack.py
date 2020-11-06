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
    check_param_type, check_param_multi_types,\
    check_value_non_negative, check_detection_inputs
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
            turns on untargeted attack. It should be noted that only untargeted attack
            is supproted for model_type='detection', Default: False.
        sparse (bool): If True, input labels are sparse-encoded. If False,
            input labels are one-hot-encoded. Default: True.
        model_type (str): The type of targeted model. 'classification' and 'detection' are supported now.
            default: 'classification'.
        reserve_ratio (float): The percentage of objects that can be detected after attacks, specifically for
            model_type='detection'. Default: 0.3.

    Examples:
        >>> attack = PSOAttack(model)
    """

    def __init__(self, model, model_type='classification', targeted=False, reserve_ratio=0.3, sparse=True,
                 step_size=0.5, per_bounds=0.6, c1=2.0, c2=2.0, c=2.0, pop_size=6, t_max=1000, pm=0.5, bounds=None):
        super(PSOAttack, self).__init__()
        self._model = check_model('model', model, BlackModel)
        self._step_size = check_value_positive('step_size', step_size)
        self._per_bounds = check_value_positive('per_bounds', per_bounds)
        self._c1 = check_value_positive('c1', c1)
        self._c2 = check_value_positive('c2', c2)
        self._c = check_value_positive('c', c)
        self._pop_size = check_int_positive('pop_size', pop_size)
        self._pm = check_value_positive('pm', pm)
        self._bounds = bounds
        if self._bounds is not None:
            self._bounds = check_param_multi_types('bounds', bounds, [list, tuple])
            for b in self._bounds:
                _ = check_param_multi_types('bound', b, [int, float])
        self._targeted = check_param_type('targeted', targeted, bool)
        self._t_max = check_int_positive('t_max', t_max)
        self._sparse = check_param_type('sparse', sparse, bool)
        self._model_type = check_param_type('model_type', model_type, str)
        if self._model_type not in ('classification', 'detection'):
            msg = "Only 'classification' or 'detection' is supported now, but got {}.".format(self._model_type)
            LOGGER.error(TAG, msg)
            raise ValueError(msg)
        self._reserve_ratio = check_value_non_negative('reserve_ratio', reserve_ratio)
        if self._reserve_ratio > 1:
            msg = "reserve_ratio should be less than 1.0, but got {}.".format(self._reserve_ratio)
            LOGGER.error(TAG, msg)
            raise ValueError(msg)

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
        if np.max(fit_value) < 0:
            self._c /= 2
        return fit_value

    def _confidence_cla(self, inputs, labels):
        """
        Calculate the prediction confidence of corresponding label or max confidence of inputs.

        Args:
            inputs (numpy.ndarray): Input samples.
            labels (Union[numpy.int, numpy.int16, numpy.int32, numpy.int64]): Target labels.

        Returns:
            float, the prediction confidences of inputs.
        """
        check_numpy_param('inputs', inputs)
        check_param_multi_types('labels', labels, (np.int, np.int16, np.int32, np.int64))
        confidences = self._model.predict(inputs)
        if self._targeted:
            confi_choose = confidences[:, labels]
        else:
            confi_choose = np.max(confidences, axis=1)
        return confi_choose

    def _mutation_op(self, cur_pop):
        """
        Generate mutation samples.

        Args:
            cur_pop (numpy.ndarray): Inputs before mutation operation.

        Returns:
            numpy.ndarray, mutational inputs.
        """
        # LOGGER.info(TAG, 'Mutation happens...')
        pixel_deep = self._bounds[1] - self._bounds[0]
        cur_pop = check_numpy_param('cur_pop', cur_pop)
        perturb_noise = (np.random.random(cur_pop.shape) - 0.5)*pixel_deep
        mutated_pop = perturb_noise + cur_pop
        if self._model_type == 'classification':
            mutated_pop = np.clip(np.clip(mutated_pop, cur_pop - self._per_bounds*np.abs(cur_pop),
                                          cur_pop + self._per_bounds*np.abs(cur_pop)),
                                  self._bounds[0], self._bounds[1])
        return mutated_pop

    def generate(self, inputs, labels):
        """
        Generate adversarial examples based on input data and targeted
        labels (or ground_truth labels).

        Args:
            inputs (Union[numpy.ndarray, tuple]): Input samples. The format of inputs can be (input1, input2, ...)
                or only one array if model_type='detection'.
            labels (Union[numpy.ndarray, tuple]): Targeted labels or ground_truth labels. The format of labels
                should be (gt_boxes, gt_labels) if model_type='detection'.

        Returns:
            - numpy.ndarray, bool values for each attack result.

            - numpy.ndarray, generated adversarial examples.

            - numpy.ndarray, query times for each sample.

        Examples:
            >>> advs = attack.generate([[0.2, 0.3, 0.4], [0.3, 0.3, 0.2]],
            >>> [1, 2])
        """
        # inputs check
        if self._model_type == 'classification':
            inputs, labels = check_pair_numpy_param('inputs', inputs,
                                                    'labels', labels)
            if not self._sparse:
                labels = np.argmax(labels, axis=1)
            images = inputs
        elif self._model_type == 'detection':
            images, auxiliary_inputs, gt_boxes, gt_labels = check_detection_inputs(inputs, labels)

        # generate one adversarial each time
        adv_list = []
        success_list = []
        query_times_list = []
        for i in range(images.shape[0]):
            is_success = False
            q_times = 0
            x_ori = images[i]
            if not self._bounds:
                self._bounds = [np.min(x_ori), np.max(x_ori)]
            pixel_deep = self._bounds[1] - self._bounds[0]

            q_times += 1
            if self._model_type == 'classification':
                label_i = labels[i]
                confi_ori = self._confidence_cla(x_ori, label_i)
            elif self._model_type == 'detection':
                auxiliary_input_i = tuple()
                for item in auxiliary_inputs:
                    auxiliary_input_i += (np.expand_dims(item[i], axis=0),)
                gt_boxes_i, gt_labels_i = gt_boxes[i], gt_labels[i]
                inputs_i = (images[i],) + auxiliary_input_i
                confi_ori, gt_object_num = self._detection_scores(inputs_i, gt_boxes_i, gt_labels_i, self._model)
                LOGGER.info(TAG, 'The number of ground-truth objects is %s', gt_object_num[0])

            # step1, initializing
            # initial global optimum fitness value, cannot set to be -inf
            best_fitness = -np.inf
            # initial global optimum position
            best_position = x_ori
            x_copies = np.repeat(x_ori[np.newaxis, :], self._pop_size, axis=0)
            cur_noise = np.clip((np.random.random(x_copies.shape) - 0.5)*pixel_deep*self._step_size,
                                (0 - self._per_bounds)*(np.abs(x_copies) + 0.1),
                                self._per_bounds*(np.abs(x_copies) + 0.1))
            par = np.clip(x_copies + cur_noise, self._bounds[0], self._bounds[1])
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
                par = np.clip(np.clip(par + v_particles,
                                      par_ori - (np.abs(par_ori) + 0.1*pixel_deep)*self._per_bounds,
                                      par_ori + (np.abs(par_ori) + 0.1*pixel_deep)*self._per_bounds),
                              self._bounds[0], self._bounds[1])
                if iters > 20 and is_mutation:
                    par = self._mutation_op(par)
                if self._model_type == 'classification':
                    confi_adv = self._confidence_cla(par, label_i)
                elif self._model_type == 'detection':
                    confi_adv, _ = self._detection_scores(
                        (par,) + auxiliary_input_i, gt_boxes_i, gt_labels_i, self._model)
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
                is_mutation = False
                if (best_fitness - last_best_fit) < last_best_fit*0.05:
                    is_mutation = True

                q_times += 1
                if self._model_type == 'classification':
                    cur_pre = self._model.predict(best_position)
                    cur_label = np.argmax(cur_pre)
                    if (self._targeted and cur_label == label_i) or (not self._targeted and cur_label != label_i):
                        is_success = True
                elif self._model_type == 'detection':
                    _, correct_nums_adv = self._detection_scores(
                        (best_position,) + auxiliary_input_i, gt_boxes_i, gt_labels_i, self._model)
                    LOGGER.info(TAG, 'The number of correctly detected objects in adversarial image is %s',
                                correct_nums_adv[0])
                    if correct_nums_adv <= int(gt_object_num*self._reserve_ratio):
                        is_success = True

                if is_success:
                    LOGGER.debug(TAG, 'successfully find one adversarial '
                                      'sample and start Reduction process')
                    # step3, reduction
                    if self._model_type == 'classification':
                        best_position, q_times = self._reduction(x_ori, q_times, label_i, best_position, self._model,
                                                                 targeted_attack=self._targeted)

                    break
            if self._model_type == 'detection':
                best_position, q_times = self._fast_reduction(x_ori, best_position, q_times,
                                                              auxiliary_input_i, gt_boxes_i, gt_labels_i, self._model)
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

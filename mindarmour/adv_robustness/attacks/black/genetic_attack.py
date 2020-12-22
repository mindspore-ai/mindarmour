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
    check_int_positive, check_detection_inputs, check_value_non_negative, check_param_multi_types
from mindarmour.adv_robustness.attacks.attack import Attack
from .black_model import BlackModel

LOGGER = LogUtil.get_instance()
TAG = 'GeneticAttack'


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
        model_type (str): The type of targeted model. 'classification' and 'detection' are supported now.
            default: 'classification'.
        targeted (bool): If True, turns on the targeted attack. If False,
            turns on untargeted attack. It should be noted that only untargeted attack
            is supproted for model_type='detection', Default: True.
        reserve_ratio (Union[int, float]): The percentage of objects that can be detected after attacks,
            specifically for model_type='detection'. Reserve_ratio should be in the range of (0, 1). Default: 0.3.
        pop_size (int): The number of particles, which should be greater than
            zero. Default: 6.
        mutation_rate (Union[int, float]): The probability of mutations, which should be in the range of (0, 1).
            Default: 0.005.
        per_bounds (Union[int, float]): Maximum L_inf distance.
        max_steps (int): The maximum round of iteration for each adversarial
            example. Default: 1000.
        step_size (Union[int, float]): Attack step size. Default: 0.2.
        temp (Union[int, float]): Sampling temperature for selection. Default: 0.3.
            The greater the temp, the greater the differences between individuals'
            selecting probabilities.
        bounds (Union[tuple, list, None]): Upper and lower bounds of data. In form
            of (clip_min, clip_max). Default: (0, 1.0).
        adaptive (bool): If True, turns on dynamic scaling of mutation
            parameters. If false, turns on static mutation parameters.
            Default: False.
        sparse (bool): If True, input labels are sparse-encoded. If False,
            input labels are one-hot-encoded. Default: True.
        c (Union[int, float]): Weight of perturbation loss. Default: 0.1.

    Examples:
        >>> attack = GeneticAttack(model)
    """
    def __init__(self, model, model_type='classification', targeted=True, reserve_ratio=0.3, sparse=True,
                 pop_size=6, mutation_rate=0.005, per_bounds=0.15, max_steps=1000, step_size=0.20, temp=0.3,
                 bounds=(0, 1.0), adaptive=False, c=0.1):
        super(GeneticAttack, self).__init__()
        self._model = check_model('model', model, BlackModel)
        self._model_type = check_param_type('model_type', model_type, str)
        if self._model_type not in ('classification', 'detection'):
            msg = "Only 'classification' or 'detection' is supported now, but got {}.".format(self._model_type)
            LOGGER.error(TAG, msg)
            raise ValueError(msg)
        self._targeted = check_param_type('targeted', targeted, bool)
        self._reserve_ratio = check_value_non_negative('reserve_ratio', reserve_ratio)
        if self._reserve_ratio > 1:
            msg = "reserve_ratio should not be greater than 1.0, but got {}.".format(self._reserve_ratio)
            LOGGER.error(TAG, msg)
            raise ValueError(msg)
        self._sparse = check_param_type('sparse', sparse, bool)
        self._per_bounds = check_value_positive('per_bounds', per_bounds)
        self._pop_size = check_int_positive('pop_size', pop_size)
        self._step_size = check_value_positive('step_size', step_size)
        self._temp = check_value_positive('temp', temp)
        self._max_steps = check_int_positive('max_steps', max_steps)
        self._mutation_rate = check_value_non_negative('mutation_rate', mutation_rate)
        if self._mutation_rate > 1:
            msg = "mutation_rate should not be greater than 1.0, but got {}.".format(self._mutation_rate)
            LOGGER.error(TAG, msg)
            raise ValueError(msg)
        self._adaptive = check_param_type('adaptive', adaptive, bool)
        # initial global optimum fitness value
        self._best_fit = -np.inf
        # count times of no progress
        self._plateau_times = 0
        # count times of changing attack step_size
        self._adap_times = 0
        self._bounds = bounds
        if self._bounds is not None:
            self._bounds = check_param_multi_types('bounds', bounds, [list, tuple])
            for b in self._bounds:
                _ = check_param_multi_types('bound', b, [int, float])
        self._c = check_value_positive('c', c)

    def _mutation(self, cur_pop, step_noise=0.01, prob=0.005):
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
                                -step_noise, step_noise)*(self._bounds[1] - self._bounds[0])
        mutated_pop = perturb_noise*(
            np.random.random(cur_pop.shape) < prob) + cur_pop
        return mutated_pop

    def generate(self, inputs, labels):
        """
        Generate adversarial examples based on input data and targeted
        labels (or ground_truth labels).

        Args:
            inputs (Union[numpy.ndarray, tuple]): Input samples. The format of inputs should be numpy.ndarray if
                model_type='classification'. The format of inputs can be (input1, input2, ...) or only one array if
                model_type='detection'.
            labels (Union[numpy.ndarray, tuple]): Targeted labels or ground-truth labels. The format of labels should
                be numpy.ndarray if model_type='classification'. The format of labels should be (gt_boxes, gt_labels)
                if model_type='detection'.

        Returns:
            - numpy.ndarray, bool values for each attack result.

            - numpy.ndarray, generated adversarial examples.

            - numpy.ndarray, query times for each sample.

        Examples:
            >>> advs = attack.generate([[0.2, 0.3, 0.4],
            >>>                         [0.3, 0.3, 0.2]],
            >>>                        [1, 2])
        """
        if self._model_type == 'classification':
            inputs, labels = check_pair_numpy_param('inputs', inputs,
                                                    'labels', labels)
            if self._sparse:
                if labels.size > 1:
                    label_squ = np.squeeze(labels)
                else:
                    label_squ = labels
                if len(label_squ.shape) >= 2 or label_squ.shape[0] != inputs.shape[0]:
                    msg = "The parameter 'sparse' of GeneticAttack is True, but the input labels is not sparse style " \
                          "and got its shape as {}.".format(labels.shape)
                    LOGGER.error(TAG, msg)
                    raise ValueError(msg)
            else:
                labels = np.argmax(labels, axis=1)
            images = inputs
        elif self._model_type == 'detection':
            images, auxiliary_inputs, gt_boxes, gt_labels = check_detection_inputs(inputs, labels)

        adv_list = []
        success_list = []
        query_times_list = []
        for i in range(images.shape[0]):
            is_success = False
            x_ori = images[i]
            if not self._bounds:
                self._bounds = [np.min(x_ori), np.max(x_ori)]
            pixel_deep = self._bounds[1] - self._bounds[0]

            if self._model_type == 'classification':
                label_i = labels[i]
            elif self._model_type == 'detection':
                auxiliary_input_i = tuple()
                for item in auxiliary_inputs:
                    auxiliary_input_i += (np.expand_dims(item[i], axis=0),)
                gt_boxes_i, gt_labels_i = np.expand_dims(gt_boxes[i], axis=0), np.expand_dims(gt_labels[i], axis=0)
                inputs_i = (images[i],) + auxiliary_input_i
                confi_ori, gt_object_num = self._detection_scores(inputs_i, gt_boxes_i, gt_labels_i, model=self._model)
                LOGGER.info(TAG, 'The number of ground-truth objects is %s', gt_object_num[0])

            # generate particles
            ori_copies = np.repeat(x_ori[np.newaxis, :], self._pop_size, axis=0)
            # initial perturbations
            cur_pert = np.random.uniform(self._bounds[0], self._bounds[1], ori_copies.shape)
            cur_pop = ori_copies + cur_pert
            query_times = 0
            iters = 0

            while iters < self._max_steps:
                iters += 1
                cur_pop = np.clip(np.clip(cur_pop,
                                          ori_copies - pixel_deep*self._per_bounds,
                                          ori_copies + pixel_deep*self._per_bounds),
                                  self._bounds[0], self._bounds[1])

                if self._model_type == 'classification':
                    pop_preds = self._model.predict(cur_pop)
                    query_times += cur_pop.shape[0]
                    all_preds = np.argmax(pop_preds, axis=1)
                    if self._targeted:
                        success_pop = np.equal(label_i, all_preds).astype(np.int32)
                    else:
                        success_pop = np.not_equal(label_i, all_preds).astype(np.int32)
                    is_success = max(success_pop)
                    best_idx = np.argmax(success_pop)
                    target_preds = pop_preds[:, label_i]
                    others_preds_sum = np.sum(pop_preds, axis=1) - target_preds
                    if self._targeted:
                        fit_vals = target_preds - others_preds_sum
                    else:
                        fit_vals = others_preds_sum - target_preds

                elif self._model_type == 'detection':
                    confi_adv, correct_nums_adv = self._detection_scores(
                        (cur_pop,) + auxiliary_input_i, gt_boxes_i, gt_labels_i, model=self._model)
                    LOGGER.info(TAG, 'The number of correctly detected objects in adversarial image is %s',
                                np.min(correct_nums_adv))
                    query_times += self._pop_size
                    fit_vals = abs(
                        confi_ori - confi_adv) - self._c / self._pop_size * np.linalg.norm(
                            (cur_pop - x_ori).reshape(cur_pop.shape[0], -1), axis=1)

                    if np.max(fit_vals) < 0:
                        self._c /= 2

                    if np.max(fit_vals) < -2:
                        LOGGER.debug(TAG,
                                     'best fitness value is %s, which is too small. We recommend that you decrease '
                                     'the value of the initialization parameter c.', np.max(fit_vals))
                    if iters < 3 and np.max(fit_vals) > 100:
                        LOGGER.debug(TAG,
                                     'best fitness value is %s, which is too large. We recommend that you increase '
                                     'the value of the initialization parameter c.', np.max(fit_vals))

                    if np.min(correct_nums_adv) <= int(gt_object_num*self._reserve_ratio):
                        is_success = True
                        best_idx = np.argmin(correct_nums_adv)

                if is_success:
                    LOGGER.debug(TAG, 'successfully find one adversarial sample '
                                      'and start Reduction process.')
                    final_adv = cur_pop[best_idx]
                    if self._model_type == 'classification':
                        final_adv, query_times = self._reduction(x_ori, query_times, label_i, final_adv,
                                                                 model=self._model, targeted_attack=self._targeted)
                    break

                best_fit = max(fit_vals)

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
                    step_p = max(self._mutation_rate, 0.5*(0.9**self._adap_times))
                else:
                    step_noise = self._step_size
                    step_p = self._mutation_rate
                step_temp = self._temp
                elite = cur_pop[np.argmax(fit_vals)]
                select_probs = softmax(fit_vals/step_temp)
                select_args = np.arange(self._pop_size)
                parents_arg = np.random.choice(
                    a=select_args, size=2*(self._pop_size - 1),
                    replace=True, p=select_probs)
                parent1 = cur_pop[parents_arg[:self._pop_size - 1]]
                parent2 = cur_pop[parents_arg[self._pop_size - 1:]]
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
                mutated_childs = self._mutation(
                    childs, step_noise=self._per_bounds*step_noise,
                    prob=step_p)
                cur_pop = np.concatenate((mutated_childs, elite[np.newaxis, :]))

            if not is_success:
                LOGGER.debug(TAG, 'fail to find adversarial sample.')
                final_adv = elite
            if self._model_type == 'detection':
                final_adv, query_times = self._fast_reduction(
                    x_ori, final_adv, query_times, auxiliary_input_i, gt_boxes_i, gt_labels_i, model=self._model)
            adv_list.append(final_adv)

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

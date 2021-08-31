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
Fuzzing.
"""
from random import choice
from copy import deepcopy
import numpy as np
from mindspore import Model
from mindspore import Tensor
from mindspore import nn

from mindarmour.utils._check_param import check_model, check_numpy_param, check_param_multi_types, check_norm_level, \
    check_param_in_range, check_param_type, check_int_positive, check_param_bounds
from mindarmour.utils.logger import LogUtil
from ..adv_robustness.attacks import FastGradientSignMethod, \
    MomentumDiverseInputIterativeMethod, ProjectedGradientDescent
from .image_transform import Contrast, Brightness, Blur, \
    Noise, Translate, Scale, Shear, Rotate
from .model_coverage_metrics import CoverageMetrics, KMultisectionNeuronCoverage

LOGGER = LogUtil.get_instance()
TAG = 'Fuzzer'


def _select_next(initial_seeds):
    """ Randomly select a seed from `initial_seeds`."""
    seed_num = choice(range(len(initial_seeds)))
    seed = initial_seeds[seed_num]
    del initial_seeds[seed_num]
    return seed, initial_seeds


def _coverage_gains(pre_coverage, coverages):
    """
    Calculate the coverage gains of mutated samples.

    Args:
        pre_coverage (float): Last value of coverages for previous mutated samples.
        coverages (list): Coverage of mutated samples.

    Returns:
        - list, coverage gains for mutated samples.

        - float, last value in parameter coverages.
    """
    gains = [pre_coverage] + coverages[:-1]
    gains = np.array(coverages) - np.array(gains)
    return gains, coverages[-1]


def _is_trans_valid(seed, mutate_sample):
    """
    Check a mutated sample is valid. If the number of changed pixels in
    a seed is less than pixels_change_rate*size(seed), this mutate is valid.
    Else check the infinite norm of seed changes, if the value of the
    infinite norm less than pixel_value_change_rate*255, this mutate is
    valid too. Otherwise the opposite.
    """
    is_valid = False
    pixels_change_rate = 0.02
    pixel_value_change_rate = 0.2
    diff = np.array(seed - mutate_sample).flatten()
    size = np.shape(diff)[0]
    l0_norm = np.linalg.norm(diff, ord=0)
    linf = np.linalg.norm(diff, ord=np.inf)
    if l0_norm > pixels_change_rate * size:
        if linf < 256:
            is_valid = True
    else:
        if linf < pixel_value_change_rate * 255:
            is_valid = True
    return is_valid


def _gain_threshold(coverage):
    """Get threshold for given neuron coverage class."""
    if coverage is isinstance(coverage, KMultisectionNeuronCoverage):
        gain_threshold = 0.1 / coverage.segmented_num
    else:
        gain_threshold = 0
    return gain_threshold


class Fuzzer:
    """
    Fuzzing test framework for deep neural networks.

    Reference: `DeepHunter: A Coverage-Guided Fuzz Testing Framework for Deep
    Neural Networks <https://dl.acm.org/doi/10.1145/3293882.3330579>`_

    Args:
        target_model (Model): Target fuzz model.

    Examples:
        >>> net = Net()
        >>> model = Model(net)
        >>> mutate_config = [{'method': 'Blur',
        >>>                   'params': {'auto_param': [True]}},
        >>>                  {'method': 'Contrast',
        >>>                   'params': {'factor': [2]}},
        >>>                  {'method': 'Translate',
        >>>                   'params': {'x_bias': [0.1, 0.2], 'y_bias': [0.2]}},
        >>>                  {'method': 'FGSM',
        >>>                   'params': {'eps': [0.1, 0.2, 0.3], 'alpha': [0.1]}}]
        >>> nc = KMultisectionNeuronCoverage(model, train_images, segmented_num=100)
        >>> model_fuzz_test = Fuzzer(model)
        >>> samples, gt_labels, preds, strategies, metrics = model_fuzz_test.fuzzing(mutate_config, initial_seeds,
        >>>                                                                          nc, max_iters=100)
    """

    def __init__(self, target_model):
        self._target_model = check_model('model', target_model, Model)

        # Allowed mutate strategies so far.
        self._strategies = {'Contrast': Contrast,
                            'Brightness': Brightness,
                            'Blur': Blur,
                            'Noise': Noise,
                            'Translate': Translate,
                            'Scale': Scale,
                            'Shear': Shear,
                            'Rotate': Rotate,
                            'FGSM': FastGradientSignMethod,
                            'PGD': ProjectedGradientDescent,
                            'MDIIM': MomentumDiverseInputIterativeMethod}
        self._affine_trans_list = ['Translate', 'Scale', 'Shear', 'Rotate']
        self._pixel_value_trans_list = ['Contrast', 'Brightness', 'Blur', 'Noise']
        self._attacks_list = ['FGSM', 'PGD', 'MDIIM']
        self._attack_param_checklists = {
            'FGSM': {'eps': {'dtype': [float], 'range': [0, 1]},
                     'alpha': {'dtype': [float], 'range': [0, 1]},
                     'bounds': {'dtype': [tuple, list]}},
            'PGD': {'eps': {'dtype': [float], 'range': [0, 1]},
                    'eps_iter': {'dtype': [float], 'range': [0, 1]},
                    'nb_iter': {'dtype': [int], 'range': [0, 100000]},
                    'bounds': {'dtype': [tuple, list]}},
            'MDIIM': {'eps': {'dtype': [float], 'range': [0, 1]},
                      'norm_level': {'dtype': [str, int], 'range': [1, 2, '1', '2', 'l1', 'l2', 'inf', 'np.inf']},
                      'prob': {'dtype': [float], 'range': [0, 1]},
                      'bounds': {'dtype': [tuple, list]}}}

    def fuzzing(self, mutate_config, initial_seeds, coverage, evaluate=True, max_iters=10000, mutate_num_per_seed=20):
        """
        Fuzzing tests for deep neural networks.

        Args:
            mutate_config (list): Mutate configs. The format is
                [{'method': 'Blur',
                'params': {'radius': [0.1, 0.2], 'auto_param': [True, False]}},
                {'method': 'Contrast',
                'params': {'factor': [1, 1.5, 2]}},
                {'method': 'FGSM',
                'params': {'eps': [0.3, 0.2, 0.4], 'alpha': [0.1]}},
                ...].
                The supported methods list is in `self._strategies`, and the params of each method must within the
                range of optional parameters. Supported methods are grouped in three types: Firstly, pixel value based
                transform methods include: 'Contrast', 'Brightness', 'Blur' and 'Noise'. Secondly, affine transform
                methods include: 'Translate', 'Scale', 'Shear' and 'Rotate'. Thirdly, attack methods include: 'FGSM',
                'PGD' and 'MDIIM'. `mutate_config` must have method in the type of pixel value based transform methods.
                The way of setting parameters for first and second type methods can be seen in
                'mindarmour/fuzz_testing/image_transform.py'. For third type methods, the optional parameters refer to
                `self._attack_param_checklists`.
            initial_seeds (list[list]): Initial seeds used to generate mutated samples. The format of initial seeds is
                [[image_data, label], [...], ...] and the label must be one-hot.
            coverage (CoverageMetrics): Class of neuron coverage metrics.
            evaluate (bool): return evaluate report or not. Default: True.
            max_iters (int): Max number of select a seed to mutate. Default: 10000.
            mutate_num_per_seed (int): The number of mutate times for a seed. Default: 20.

        Returns:
            - list, mutated samples in fuzz_testing.

            - list, ground truth labels of mutated samples.

            - list, preds of mutated samples.

            - list, strategies of mutated samples.

            - dict, metrics report of fuzzer.

        Raises:
            ValueError: Coverage must be subclass of CoverageMetrics.
            ValueError: If initial seeds is empty.
            ValueError: If element of seed is not two in initial seeds.
        """
        # Check parameters.
        if not isinstance(coverage, CoverageMetrics):
            msg = 'coverage must be subclass of CoverageMetrics'
            LOGGER.error(TAG, msg)
            raise ValueError(msg)
        evaluate = check_param_type('evaluate', evaluate, bool)
        max_iters = check_int_positive('max_iters', max_iters)
        mutate_num_per_seed = check_int_positive('mutate_num_per_seed', mutate_num_per_seed)
        mutate_config = self._check_mutate_config(mutate_config)
        mutates = self._init_mutates(mutate_config)

        initial_seeds = check_param_type('initial_seeds', initial_seeds, list)
        init_seeds = deepcopy(initial_seeds)
        if not init_seeds:
            msg = 'initial_seeds must not be empty.'
            raise ValueError(msg)
        initial_samples = []
        for seed in init_seeds:
            check_param_type('seed', seed, list)
            if len(seed) != 2:
                msg = 'seed in initial seeds must have two element image and label, but got {} element.'.format(
                    len(seed))
                raise ValueError(msg)
            check_numpy_param('seed[0]', seed[0])
            check_numpy_param('seed[1]', seed[1])
            initial_samples.append(seed[0])
            seed.append(0)
        initial_samples = np.array(initial_samples)
        # calculate the coverage of initial seeds
        pre_coverage = coverage.get_metrics(initial_samples)
        gain_threshold = _gain_threshold(coverage)

        seed, init_seeds = _select_next(init_seeds)
        fuzz_samples = []
        true_labels = []
        fuzz_preds = []
        fuzz_strategies = []
        iter_num = 0
        while init_seeds and iter_num < max_iters:
            # Mutate a seed.
            mutate_samples, mutate_strategies = self._metamorphic_mutate(seed, mutates, mutate_config,
                                                                         mutate_num_per_seed)
            # Calculate the coverages and predictions of generated samples.
            coverages, predicts = self._get_coverages_and_predict(mutate_samples, coverage)
            coverage_gains, pre_coverage = _coverage_gains(pre_coverage, coverages)
            for mutate, cov, pred, strategy in zip(mutate_samples, coverage_gains, predicts, mutate_strategies):
                fuzz_samples.append(mutate[0])
                true_labels.append(mutate[1])
                fuzz_preds.append(pred)
                fuzz_strategies.append(strategy)
                # if the mutate samples has coverage gains add this samples in the initial_seeds to guide new mutates.
                if cov > gain_threshold:
                    init_seeds.append(mutate)
            seed, init_seeds = _select_next(init_seeds)
            iter_num += 1
        metrics_report = None
        if evaluate:
            metrics_report = self._evaluate(fuzz_samples, true_labels, fuzz_preds, fuzz_strategies, coverage)
        return fuzz_samples, true_labels, fuzz_preds, fuzz_strategies, metrics_report

    def _get_coverages_and_predict(self, mutate_samples, coverage):
        """ Calculate the coverages and predictions of generated samples."""
        samples = [s[0] for s in mutate_samples]
        samples = np.array(samples)
        coverages = []
        predictions = self._target_model.predict(Tensor(samples.astype(np.float32)))
        predictions = predictions.asnumpy()
        for index in range(len(samples)):
            mutate = samples[:index + 1]
            coverages.append(coverage.get_metrics(mutate))
        return coverages, predictions

    def _metamorphic_mutate(self, seed, mutates, mutate_config, mutate_num_per_seed):
        """Mutate a seed using strategies random selected from mutate_config."""
        mutate_samples = []
        mutate_strategies = []
        for _ in range(mutate_num_per_seed):
            only_pixel_trans = seed[2]
            strategy = choice(mutate_config)
            # Choose a pixel value based transform method
            if only_pixel_trans:
                while strategy['method'] not in self._pixel_value_trans_list:
                    strategy = choice(mutate_config)
            transform = mutates[strategy['method']]
            params = strategy['params']
            method = strategy['method']
            selected_param = {}
            for param in params:
                selected_param[param] = choice(params[param])

            if method in list(self._pixel_value_trans_list + self._affine_trans_list):
                if method == 'Shear':
                    shear_keys = selected_param.keys()
                    if 'factor_x' in shear_keys and 'factor_y' in shear_keys:
                        selected_param[choice(['factor_x', 'factor_y'])] = 0
                transform.set_params(**selected_param)
                mutate_sample = transform.transform(seed[0])
            else:
                for param_name in selected_param:
                    transform.__setattr__('_' + str(param_name), selected_param[param_name])
                mutate_sample = transform.generate(np.array([seed[0].astype(np.float32)]), np.array([seed[1]]))[0]
            if method not in self._pixel_value_trans_list:
                only_pixel_trans = 1
            mutate_sample = [mutate_sample, seed[1], only_pixel_trans]
            if _is_trans_valid(seed[0], mutate_sample[0]):
                mutate_samples.append(mutate_sample)
                mutate_strategies.append(method)
        if not mutate_samples:
            mutate_samples.append(seed)
            mutate_strategies.append(None)
        return np.array(mutate_samples), mutate_strategies

    def _check_mutate_config(self, mutate_config):
        """Check whether the mutate_config meet the specification."""
        mutate_config = check_param_type('mutate_config', mutate_config, list)
        has_pixel_trans = False

        for config in mutate_config:
            check_param_type("config", config, dict)
            if set(config.keys()) != {'method', 'params'}:
                msg = "The key of each config must be in ('method', 'params'), but got {}.".format(set(config.keys()))
                LOGGER.error(TAG, msg)
                raise KeyError(msg)

            method = config['method']
            params = config['params']

            # Method must be in the optional range.
            if method not in self._strategies.keys():
                msg = "Config methods must be in {}, but got {}.".format(self._strategies.keys(), method)
                LOGGER.error(TAG, msg)
                raise ValueError(msg)

            if config['method'] in self._pixel_value_trans_list:
                has_pixel_trans = True

            check_param_type('params', params, dict)
            # Check parameters of attack methods. The parameters of transformed
            # methods will be verified in transferred parameters.
            if method in self._attacks_list:
                self._check_attack_params(method, params)
            else:
                for key in params.keys():
                    check_param_type(str(key), params[key], list)
        # Methods in `metate_config` should at least have one in the type of pixel value based transform methods.
        if not has_pixel_trans:
            msg = "mutate methods in mutate_config should at least have one in {}".format(self._pixel_value_trans_list)
            raise ValueError(msg)

        return mutate_config

    def _check_attack_params(self, method, params):
        """Check input parameters of attack methods."""
        allow_params = self._attack_param_checklists[method].keys()
        for param_name in params:
            if param_name not in allow_params:
                msg = "parameters of {} must in {}".format(method, allow_params)
                raise ValueError(msg)

            check_param_type(param_name, params[param_name], list)
            for param_value in params[param_name]:
                if param_name == 'bounds':
                    _ = check_param_bounds('bounds', param_value)
                elif param_name == 'norm_level':
                    _ = check_norm_level(param_value)
                else:
                    allow_type = self._attack_param_checklists[method][param_name]['dtype']
                    allow_range = self._attack_param_checklists[method][param_name]['range']
                    _ = check_param_multi_types(str(param_name), param_value, allow_type)
                    _ = check_param_in_range(str(param_name), param_value, allow_range[0], allow_range[1])

    def _init_mutates(self, mutate_config):
        """ Check whether the mutate_config meet the specification."""
        mutates = {}
        for mutate in mutate_config:
            method = mutate['method']
            if method not in self._attacks_list:
                mutates[method] = self._strategies[method]()
            else:
                network = self._target_model._network
                loss_fn = self._target_model._loss_fn
                if loss_fn is None:
                    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=False)
                mutates[method] = self._strategies[method](network, loss_fn=loss_fn)
        return mutates

    def _evaluate(self, fuzz_samples, true_labels, fuzz_preds, fuzz_strategies, coverage):
        """
        Evaluate generated fuzz_testing samples in three dimensions: accuracy, attack success rate and neural coverage.

        Args:
            fuzz_samples ([numpy.ndarray, list]): Generated fuzz_testing samples according to seeds.
            true_labels ([numpy.ndarray, list]): Ground truth labels of seeds.
            fuzz_preds ([numpy.ndarray, list]): Predictions of generated fuzz samples.
            fuzz_strategies ([numpy.ndarray, list]): Mutate strategies of fuzz samples.
            coverage (CoverageMetrics): Neuron coverage metrics class.

        Returns:
            dict, evaluate metrics include accuracy, attack success rate and neural coverage.
        """
        fuzz_samples = np.array(fuzz_samples)
        true_labels = np.asarray(true_labels)
        fuzz_preds = np.asarray(fuzz_preds)
        temp = np.argmax(true_labels, axis=1) == np.argmax(fuzz_preds, axis=1)
        metrics_report = {}

        if temp.any():
            acc = np.sum(temp) / np.size(temp)
        else:
            acc = 0
        metrics_report['Accuracy'] = acc

        cond = [elem in self._attacks_list for elem in fuzz_strategies]
        temp = temp[cond]
        if temp.any():
            attack_success_rate = 1 - np.sum(temp) / np.size(temp)
        else:
            attack_success_rate = None
        metrics_report['Attack_success_rate'] = attack_success_rate

        metrics_report['Coverage_metrics'] = coverage.get_metrics(fuzz_samples)

        return metrics_report

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

import numpy as np
from mindspore import Model
from mindspore import Tensor

from mindarmour.utils._check_param import check_model, check_numpy_param, \
    check_param_multi_types, check_norm_level, check_param_in_range, \
    check_param_type, check_int_positive
from mindarmour.utils.logger import LogUtil
from ..adv_robustness.attacks import FastGradientSignMethod, \
    MomentumDiverseInputIterativeMethod, ProjectedGradientDescent
from .image_transform import Contrast, Brightness, Blur, \
    Noise, Translate, Scale, Shear, Rotate
from .model_coverage_metrics import ModelCoverageMetrics

LOGGER = LogUtil.get_instance()
TAG = 'Fuzzer'


def _select_next(initial_seeds):
    """ Randomly select a seed from `initial_seeds`."""
    seed_num = choice(range(len(initial_seeds)))
    seed = initial_seeds[seed_num]
    del initial_seeds[seed_num]
    return seed, initial_seeds


def _coverage_gains(coverages):
    """ Calculate the coverage gains of mutated samples. """
    gains = [0] + coverages[:-1]
    gains = np.array(coverages) - np.array(gains)
    return gains


def _is_trans_valid(seed, mutate_sample):
    """ Check a mutated sample is valid. If the number of changed pixels in
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
    if l0_norm > pixels_change_rate*size:
        if linf < 256:
            is_valid = True
    else:
        if linf < pixel_value_change_rate*255:
            is_valid = True
    return is_valid


class Fuzzer:
    """
    Fuzzing test framework for deep neural networks.

    Reference: `DeepHunter: A Coverage-Guided Fuzz Testing Framework for Deep
    Neural Networks <https://dl.acm.org/doi/10.1145/3293882.3330579>`_

    Args:
        target_model (Model): Target fuzz model.
        train_dataset (numpy.ndarray): Training dataset used for determining
            the neurons' output boundaries.
        neuron_num (int): The number of testing neurons.
        segmented_num (int): The number of segmented sections of neurons'
            output intervals. Default: 1000.

    Examples:
        >>> net = Net()
        >>> mutate_config = [{'method': 'Blur', 'params': {'auto_param': True}},
        >>>                  {'method': 'Contrast','params': {'factor': 2}},
        >>>                  {'method': 'Translate', 'params': {'x_bias': 0.1, 'y_bias': 0.2}},
        >>>                  {'method': 'FGSM', 'params': {'eps': 0.1, 'alpha': 0.1}}]
        >>> train_images = np.random.rand(32, 1, 32, 32).astype(np.float32)
        >>> model_fuzz_test = Fuzzer(model, train_images, 10, 1000)
        >>> samples, labels, preds, strategies, report = model_fuzz_test.fuzz_testing(mutate_config, initial_seeds)
    """

    def __init__(self, target_model, train_dataset, neuron_num, segmented_num=1000):
        self._target_model = check_model('model', target_model, Model)
        train_dataset = check_numpy_param('train_dataset', train_dataset)
        self._coverage_metrics = ModelCoverageMetrics(target_model,
                                                      neuron_num,
                                                      segmented_num,
                                                      train_dataset)
        # Allowed mutate strategies so far.
        self._strategies = {'Contrast': Contrast, 'Brightness': Brightness,
                            'Blur': Blur, 'Noise': Noise, 'Translate': Translate,
                            'Scale': Scale, 'Shear': Shear, 'Rotate': Rotate,
                            'FGSM': FastGradientSignMethod,
                            'PGD': ProjectedGradientDescent,
                            'MDIIM': MomentumDiverseInputIterativeMethod}
        self._affine_trans_list = ['Translate', 'Scale', 'Shear', 'Rotate']
        self._pixel_value_trans_list = ['Contrast', 'Brightness', 'Blur',
                                        'Noise']
        self._attacks_list = ['FGSM', 'PGD', 'MDIIM']
        self._attack_param_checklists = {
            'FGSM': {'params': {'eps': {'dtype': [float], 'range': [0, 1]},
                                'alpha': {'dtype': [float],
                                          'range': [0, 1]},
                                'bounds': {'dtype': [tuple]}}},
            'PGD': {'params': {'eps': {'dtype': [float], 'range': [0, 1]},
                               'eps_iter': {'dtype': [float],
                                            'range': [0, 1]},
                               'nb_iter': {'dtype': [int],
                                           'range': [0, 1e5]},
                               'bounds': {'dtype': [tuple]}}},
            'MDIIM': {
                'params': {'eps': {'dtype': [float], 'range': [0, 1]},
                           'norm_level': {'dtype': [str]},
                           'prob': {'dtype': [float], 'range': [0, 1]},
                           'bounds': {'dtype': [tuple]}}}}

    def fuzzing(self, mutate_config, initial_seeds, coverage_metric='KMNC',
                eval_metrics='auto', max_iters=10000, mutate_num_per_seed=20):
        """
        Fuzzing tests for deep neural networks.

        Args:
            mutate_config (list): Mutate configs. The format is
                [{'method': 'Blur', 'params': {'auto_param': True}},
                {'method': 'Contrast', 'params': {'factor': 2}}]. The
                supported methods list is in `self._strategies`, and the
                params of each method must within the range of changeable parameters.　
                Supported methods are grouped in three types:
                Firstly, pixel value based transform methods include:
                'Contrast', 'Brightness', 'Blur' and 'Noise'. Secondly, affine
                transform methods include: 'Translate', 'Scale', 'Shear' and
                'Rotate'. Thirdly, attack methods include: 'FGSM', 'PGD' and 'MDIIM'.
                `mutate_config` must have method in the type of pixel value based
                transform methods. The way of setting parameters for first and
                second type methods can be seen in 'mindarmour/fuzz_testing/image_transform.py'.
                For third type methods, you can refer to the corresponding class.
            initial_seeds (list[list]): Initial seeds used to generate mutated
                samples. The format of initial seeds is [[image_data, label],
                [...], ...].
            coverage_metric (str): Model coverage metric of neural networks. All
                supported metrics are: 'KMNC', 'NBC', 'SNAC'. Default: 'KMNC'.
            eval_metrics (Union[list, tuple, str]): Evaluation metrics. If the
                type is 'auto', it will calculate all the metrics, else if the
                type is list or tuple, it will calculate the metrics specified
                by user. All supported evaluate methods are 'accuracy',
                'attack_success_rate', 'kmnc', 'nbc', 'snac'. Default: 'auto'.
            max_iters (int): Max number of select a seed to mutate.
                Default: 10000.
            mutate_num_per_seed (int): The number of mutate times for a seed.
                Default: 20.

        Returns:
            - list, mutated samples in fuzz_testing.

            - list, ground truth labels of mutated samples.

            - list, preds of mutated samples.

            - list, strategies of mutated samples.

            - dict, metrics report of fuzzer.

        Raises:
            TypeError: If the type of `eval_metrics` is not str, list or tuple.
            TypeError: If the type of metric in list `eval_metrics` is not str.
            ValueError: If `eval_metrics` is not equal to 'auto' when it's type is str.
            ValueError: If metric in list `eval_metrics` is not in ['accuracy', 'attack_success_rate',
                'kmnc', 'nbc', 'snac'].
        """
        if isinstance(eval_metrics, (list, tuple)):
            eval_metrics_ = []
            avaliable_metrics = ['accuracy', 'attack_success_rate', 'kmnc', 'nbc', 'snac']
            for elem in eval_metrics:
                if elem not in avaliable_metrics:
                    msg = 'metric in list `eval_metrics` must be in {}, but got {}.' \
                        .format(avaliable_metrics, elem)
                    LOGGER.error(TAG, msg)
                    raise ValueError(msg)
                eval_metrics_.append(elem.lower())
        elif isinstance(eval_metrics, str):
            if eval_metrics != 'auto':
                msg = "the value of `eval_metrics` must be 'auto' if it's type is str, " \
                      "but got {}.".format(eval_metrics)
                LOGGER.error(TAG, msg)
                raise ValueError(msg)
            eval_metrics_ = 'auto'
        else:
            msg = "the type of `eval_metrics` must be str, list or tuple, but got {}." \
                .format(type(eval_metrics))
            LOGGER.error(TAG, msg)
            raise TypeError(msg)

        # Check whether the mutate_config meet the specification.
        mutate_config = check_param_type('mutate_config', mutate_config, list)
        for config in mutate_config:
            check_param_type("config['params']", config['params'], dict)
            if set(config.keys()) != {'method', 'params'}:
                msg = "Config must contain 'method' and 'params', but got {}." \
                    .format(set(config.keys()))
                LOGGER.error(TAG, msg)
                raise TypeError(msg)
            if config['method'] not in self._strategies.keys():
                msg = "Config methods must be in {}, but got {}." \
                    .format(self._strategies.keys(), config['method'])
                LOGGER.error(TAG, msg)
                raise TypeError(msg)
        if coverage_metric not in ['KMNC', 'NBC', 'SNAC']:
            msg = "coverage_metric must be in ['KMNC', 'NBC', 'SNAC'], but got {}." \
                .format(coverage_metric)
            LOGGER.error(TAG, msg)
            raise ValueError(msg)
        max_iters = check_int_positive('max_iters', max_iters)
        mutate_num_per_seed = check_int_positive('mutate_num_per_seed', mutate_num_per_seed)
        mutates = self._init_mutates(mutate_config)
        initial_seeds = check_param_type('initial_seeds', initial_seeds, list)
        for seed in initial_seeds:
            check_param_type('seed', seed, list)
            check_numpy_param('seed[0]', seed[0])
            check_numpy_param('seed[1]', seed[1])
            seed.append(0)
        seed, initial_seeds = _select_next(initial_seeds)
        fuzz_samples = []
        gt_labels = []
        fuzz_preds = []
        fuzz_strategies = []
        iter_num = 0
        while initial_seeds and iter_num < max_iters:
            # Mutate a seed.
            mutate_samples, mutate_strategies = self._metamorphic_mutate(seed,
                                                                         mutates,
                                                                         mutate_config,
                                                                         mutate_num_per_seed)
            # Calculate the coverages and predictions of generated samples.
            coverages, predicts = self._run(mutate_samples, coverage_metric)
            coverage_gains = _coverage_gains(coverages)
            for mutate, cov, pred, strategy in zip(mutate_samples,
                                                   coverage_gains,
                                                   predicts, mutate_strategies):
                fuzz_samples.append(mutate[0])
                gt_labels.append(mutate[1])
                fuzz_preds.append(pred)
                fuzz_strategies.append(strategy)
                # if the mutate samples has coverage gains add this samples in
                # the initial seeds to guide new mutates.
                if cov > 0:
                    initial_seeds.append(mutate)
            seed, initial_seeds = _select_next(initial_seeds)
            iter_num += 1
        metrics_report = None
        if eval_metrics_ is not None:
            metrics_report = self._evaluate(fuzz_samples, gt_labels, fuzz_preds,
                                            fuzz_strategies, eval_metrics_)
        return fuzz_samples, gt_labels, fuzz_preds, fuzz_strategies, metrics_report

    def _run(self, mutate_samples, coverage_metric="KNMC"):
        """ Calculate the coverages and predictions of generated samples."""
        samples = [s[0] for s in mutate_samples]
        samples = np.array(samples)
        coverages = []
        predictions = self._target_model.predict(Tensor(samples.astype(np.float32)))
        predictions = predictions.asnumpy()
        for index in range(len(samples)):
            mutate = samples[:index + 1]
            self._coverage_metrics.calculate_coverage(mutate.astype(np.float32))
            if coverage_metric == 'KMNC':
                coverages.append(self._coverage_metrics.get_kmnc())
            if coverage_metric == 'NBC':
                coverages.append(self._coverage_metrics.get_nbc())
            if coverage_metric == 'SNAC':
                coverages.append(self._coverage_metrics.get_snac())
        return coverages, predictions

    def _check_attack_params(self, method, params):
        """Check input parameters of attack methods."""
        allow_params = self._attack_param_checklists[method]['params'].keys()
        for param_name in params:
            if param_name not in allow_params:
                msg = "parameters of {} must in {}".format(method, allow_params)
                raise ValueError(msg)

            param_value = params[param_name]
            if param_name == 'bounds':
                bounds = check_param_multi_types('bounds', param_value,
                                                 [list, tuple])
                for bound_value in bounds:
                    _ = check_param_multi_types('bound', bound_value, [int, float])
            elif param_name == 'norm_level':
                _ = check_norm_level(param_value)
            else:
                allow_type = self._attack_param_checklists[method]['params'][param_name][
                    'dtype']
                allow_range = self._attack_param_checklists[method]['params'][param_name][
                    'range']
                _ = check_param_multi_types(str(param_name), param_value, allow_type)
                _ = check_param_in_range(str(param_name), param_value, allow_range[0],
                                         allow_range[1])

    def _metamorphic_mutate(self, seed, mutates, mutate_config,
                            mutate_num_per_seed):
        """Mutate a seed using strategies random selected from mutate_config."""
        mutate_samples = []
        mutate_strategies = []
        only_pixel_trans = seed[2]
        for _ in range(mutate_num_per_seed):
            strage = choice(mutate_config)
            # Choose a pixel value based transform method
            if only_pixel_trans:
                while strage['method'] not in self._pixel_value_trans_list:
                    strage = choice(mutate_config)
            transform = mutates[strage['method']]
            params = strage['params']
            method = strage['method']
            if method in list(self._pixel_value_trans_list + self._affine_trans_list):
                transform.set_params(**params)
                mutate_sample = transform.transform(seed[0])
            else:
                for param_name in params:
                    transform.__setattr__('_' + str(param_name), params[param_name])
                mutate_sample = transform.generate([seed[0].astype(np.float32)],
                                                   [seed[1]])[0]
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

    def _init_mutates(self, mutate_config):
        """ Check whether the mutate_config meet the specification."""
        has_pixel_trans = False
        for mutate in mutate_config:
            if mutate['method'] in self._pixel_value_trans_list:
                has_pixel_trans = True
                break
        if not has_pixel_trans:
            msg = "mutate methods in mutate_config at lease have one in {}".format(
                self._pixel_value_trans_list)
            raise ValueError(msg)
        mutates = {}
        for mutate in mutate_config:
            method = mutate['method']
            params = mutate['params']
            if method not in self._attacks_list:
                mutates[method] = self._strategies[method]()
            else:
                self._check_attack_params(method, params)
                network = self._target_model._network
                loss_fn = self._target_model._loss_fn
                mutates[method] = self._strategies[method](network,
                                                           loss_fn=loss_fn)
        return mutates

    def _evaluate(self, fuzz_samples, gt_labels, fuzz_preds,
                  fuzz_strategies, metrics):
        """
        Evaluate generated fuzz_testing samples in three dimention: accuracy,
        attack success rate and neural coverage.

        Args:
            fuzz_samples (numpy.ndarray): Generated fuzz_testing samples according to seeds.
            gt_labels (numpy.ndarray): Ground Truth of seeds.
            fuzz_preds (numpy.ndarray): Predictions of generated fuzz samples.
            fuzz_strategies (numpy.ndarray): Mutate strategies of fuzz samples.
            metrics (Union[list, tuple, str]): evaluation metrics.

        Returns:
            dict, evaluate metrics include accuarcy, attack success rate
                and neural coverage.
        """
        gt_labels = np.asarray(gt_labels)
        fuzz_preds = np.asarray(fuzz_preds)
        temp = np.argmax(gt_labels, axis=1) == np.argmax(fuzz_preds, axis=1)
        metrics_report = {}
        if metrics == 'auto' or 'accuracy' in metrics:
            if temp.any():
                acc = np.sum(temp) / np.size(temp)
            else:
                acc = 0
            metrics_report['Accuracy'] = acc

        if metrics == 'auto' or 'attack_success_rate' in metrics:
            cond = [elem in self._attacks_list for elem in fuzz_strategies]
            temp = temp[cond]
            if temp.any():
                attack_success_rate = 1 - np.sum(temp) / np.size(temp)
            else:
                attack_success_rate = None
            metrics_report['Attack_success_rate'] = attack_success_rate

        if metrics == 'auto' or 'kmnc' in metrics or 'nbc' in metrics or 'snac' in metrics:
            self._coverage_metrics.calculate_coverage(
                np.array(fuzz_samples).astype(np.float32))

        if metrics == 'auto' or 'kmnc' in metrics:
            kmnc = self._coverage_metrics.get_kmnc()
            metrics_report['Neural_coverage_KMNC'] = kmnc

        if metrics == 'auto' or 'nbc' in metrics:
            nbc = self._coverage_metrics.get_nbc()
            metrics_report['Neural_coverage_NBC'] = nbc

        if metrics == 'auto' or 'snac' in metrics:
            snac = self._coverage_metrics.get_snac()
            metrics_report['Neural_coverage_SNAC'] = snac

        return metrics_report

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

from mindarmour.fuzzing.model_coverage_metrics import ModelCoverageMetrics
from mindarmour.utils._check_param import check_model, check_numpy_param, \
    check_param_multi_types, check_norm_level, check_param_in_range
from mindarmour.fuzzing.image_transform import Contrast, Brightness, Blur, \
    Noise, Translate, Scale, Shear, Rotate
from mindarmour.attacks import FastGradientSignMethod, \
    MomentumDiverseInputIterativeMethod, ProjectedGradientDescent


class Fuzzer:
    """
    Fuzzing test framework for deep neural networks.

    Reference: `DeepHunter: A Coverage-Guided Fuzz Testing Framework for Deep
    Neural Networks <https://dl.acm.org/doi/10.1145/3293882.3330579>`_

    Args:
        target_model (Model): Target fuzz model.
        train_dataset (numpy.ndarray): Training dataset used for determining
            the neurons' output boundaries.
        segmented_num (int): The number of segmented sections of neurons'
            output intervals.
        neuron_num (int): The number of testing neurons.
    """

    def __init__(self, target_model, train_dataset, segmented_num, neuron_num):
        self.target_model = check_model('model', target_model, Model)
        self.train_dataset = check_numpy_param('train_dataset', train_dataset)
        self.coverage_metrics = ModelCoverageMetrics(target_model,
                                                     segmented_num,
                                                     neuron_num, train_dataset)
        # Allowed mutate strategies so far.
        self.strategies = {'Contrast': Contrast, 'Brightness': Brightness,
                           'Blur': Blur, 'Noise': Noise, 'Translate': Translate,
                           'Scale': Scale, 'Shear': Shear, 'Rotate': Rotate,
                           'FGSM': FastGradientSignMethod,
                           'PGD': ProjectedGradientDescent,
                           'MDIIM': MomentumDiverseInputIterativeMethod}
        self.affine_trans_list = ['Translate', 'Scale', 'Shear', 'Rotate']
        self.pixel_value_trans_list = ['Contrast', 'Brightness', 'Blur',
                                       'Noise']
        self.attacks_list = ['FGSM', 'PGD', 'MDIIM']
        self.attack_param_checklists = {
            'FGSM': {'params': {'eps': {'dtype': [float, int], 'range': [0, 1]},
                                'alpha': {'dtype': [float, int],
                                          'range': [0, 1]},
                                'bounds': {'dtype': [list, tuple],
                                           'range': None},
                                }},
            'PGD': {'params': {'eps': {'dtype': [float, int], 'range': [0, 1]},
                               'eps_iter': {'dtype': [float, int],
                                            'range': [0, 1e5]},
                               'nb_iter': {'dtype': [float, int],
                                           'range': [0, 1e5]},
                               'bounds': {'dtype': [list, tuple],
                                          'range': None},
                               }},
            'MDIIM': {
                'params': {'eps': {'dtype': [float, int], 'range': [0, 1]},
                           'norm_level': {'dtype': [str], 'range': None},
                           'prob': {'dtype': [float, int], 'range': [0, 1]},
                           'bounds': {'dtype': [list, tuple], 'range': None},
                           }}}

    def _check_attack_params(self, method, params):
        """Check input parameters of attack methods."""
        allow_params = self.attack_param_checklists[method]['params'].keys()
        for p in params:
            if p not in allow_params:
                msg = "parameters of {} must in {}".format(method, allow_params)
                raise ValueError(msg)
            if p == 'bounds':
                bounds = check_param_multi_types('bounds', params[p],
                                                 [list, tuple])
                for b in bounds:
                    _ = check_param_multi_types('bound', b, [int, float])
            elif p == 'norm_level':
                _ = check_norm_level(params[p])
            else:
                allow_type = self.attack_param_checklists[method]['params'][p][
                    'dtype']
                allow_range = self.attack_param_checklists[method]['params'][p][
                    'range']
                _ = check_param_multi_types(str(p), params[p], allow_type)
                _ = check_param_in_range(str(p), params[p], allow_range[0],
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
                while strage['method'] not in self.pixel_value_trans_list:
                    strage = choice(mutate_config)
            transform = mutates[strage['method']]
            params = strage['params']
            method = strage['method']
            if method in list(self.pixel_value_trans_list + self.affine_trans_list):
                transform.set_params(**params)
                mutate_sample = transform.transform(seed[0])
            else:
                for p in params:
                    transform.__setattr__('_'+str(p), params[p])
                mutate_sample = transform.generate([seed[0].astype(np.float32)],
                                                   [seed[1]])[0]
            if method not in self.pixel_value_trans_list:
                only_pixel_trans = 1
            mutate_sample = [mutate_sample, seed[1], only_pixel_trans]
            if self._is_trans_valid(seed[0], mutate_sample[0]):
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
            if mutate['method'] in self.pixel_value_trans_list:
                has_pixel_trans = True
                break
        if not has_pixel_trans:
            msg = "mutate methods in mutate_config at lease have one in {}".format(
                self.pixel_value_trans_list)
            raise ValueError(msg)
        mutates = {}
        for mutate in mutate_config:
            method = mutate['method']
            params = mutate['params']
            if method not in self.attacks_list:
                mutates[method] = self.strategies[method]()
            else:
                self._check_attack_params(method, params)
                network = self.target_model._network
                loss_fn = self.target_model._loss_fn
                mutates[method] = self.strategies[method](network,
                                                          loss_fn=loss_fn)
        return mutates

    def evaluate(self, fuzz_samples, gt_labels, fuzz_preds,
                 fuzz_strategies):
        """
        Evaluate generated fuzzing samples in three dimention: accuracy,
        attack success rate and neural coverage.

        Args:
            fuzz_samples (numpy.ndarray): Generated fuzzing samples according to seeds.
            gt_labels (numpy.ndarray): Ground Truth of seeds.
            fuzz_preds (numpy.ndarray): Predictions of generated fuzz samples.
            fuzz_strategies (numpy.ndarray): Mutate strategies of fuzz samples.

        Returns:
            dict, evaluate metrics include accuarcy, attack success rate
                and neural coverage.
        """

        gt_labels = np.asarray(gt_labels)
        fuzz_preds = np.asarray(fuzz_preds)
        temp = np.argmax(gt_labels, axis=1) == np.argmax(fuzz_preds, axis=1)
        acc = np.sum(temp) / np.size(temp)

        cond = [elem in self.attacks_list for elem in fuzz_strategies]
        temp = temp[cond]
        attack_success_rate = 1 - np.sum(temp) / np.size(temp)

        self.coverage_metrics.calculate_coverage(
            np.array(fuzz_samples).astype(np.float32))
        kmnc = self.coverage_metrics.get_kmnc()
        nbc = self.coverage_metrics.get_nbc()
        snac = self.coverage_metrics.get_snac()

        metrics = {}
        metrics['Accuracy'] = acc
        metrics['Attack_succrss_rate'] = attack_success_rate
        metrics['Neural_coverage_KMNC'] = kmnc
        metrics['Neural_coverage_NBC'] = nbc
        metrics['Neural_coverage_SNAC'] = snac
        return metrics

    def fuzzing(self, mutate_config, initial_seeds, coverage_metric='KMNC',
                eval_metric=True, max_iters=10000, mutate_num_per_seed=20):
        """
        Fuzzing tests for deep neural networks.

        Args:
            mutate_config (list): Mutate configs. The format is
                [{'method': 'Blur',
                  'params': {'auto_param': True}},
                 {'method': 'Contrast',
                  'params': {'factor': 2}},
                 ...]. The support methods list is in `self.strategies`,
                 The params of each method must within the range of changeable
                 parameters.
            initial_seeds (numpy.ndarray): Initial seeds used to generate
                mutated samples.
            coverage_metric (str): Model coverage metric of neural networks.
                Default: 'KMNC'.
            eval_metric (bool): Whether to evaluate the generated fuzz samples.
                Default: True.
            max_iters (int): Max number of select a seed to mutate.
                Default: 10000.
            mutate_num_per_seed (int): The number of mutate times for a seed.
                Default: 20.

        Returns:
            list, mutated samples.
        """
        # Check whether the mutate_config meet the specification.
        mutates = self._init_mutates(mutate_config)
        seed, initial_seeds = self._select_next(initial_seeds)
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
            coverage_gains = self._coverage_gains(coverages)
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
            seed, initial_seeds = self._select_next(initial_seeds)
            iter_num += 1
        metrics = None
        if eval_metric:
            metrics = self.evaluate(fuzz_samples, gt_labels, fuzz_preds,
                                    fuzz_strategies)
        return fuzz_samples, gt_labels, fuzz_preds, fuzz_strategies, metrics

    def _coverage_gains(self, coverages):
        """ Calculate the coverage gains of mutated samples. """
        gains = [0] + coverages[:-1]
        gains = np.array(coverages) - np.array(gains)
        return gains

    def _run(self, mutate_samples, coverage_metric="KNMC"):
        """ Calculate the coverages and predictions of generated samples."""
        samples = [s[0] for s in mutate_samples]
        samples = np.array(samples)
        coverages = []
        predictions = self.target_model.predict(Tensor(samples.astype(np.float32)))
        predictions = predictions.asnumpy()
        for index in range(len(samples)):
            mutate = samples[:index + 1]
            self.coverage_metrics.calculate_coverage(mutate.astype(np.float32))
            if coverage_metric == "KMNC":
                coverages.append(self.coverage_metrics.get_kmnc())
            if coverage_metric == 'NBC':
                coverages.append(self.coverage_metrics.get_nbc())
            if coverage_metric == 'SNAC':
                coverages.append(self.coverage_metrics.get_snac())
        return coverages, predictions

    def _select_next(self, initial_seeds):
        """Randomly select a seed from `initial_seeds`."""
        seed_num = choice(range(len(initial_seeds)))
        seed = initial_seeds[seed_num]
        del initial_seeds[seed_num]
        return seed, initial_seeds

    def _is_trans_valid(self, seed, mutate_sample):
        """ Check a mutated sample is valid. If the number of changed pixels in
        a seed is less than pixels_change_rate*size(seed), this mutate is valid.
        Else check the infinite norm of seed changes, if the value of the
        infinite norm less than pixel_value_change_rate*255, this mutate is
        valid too. Otherwise the opposite."""
        is_valid = False
        pixels_change_rate = 0.02
        pixel_value_change_rate = 0.2
        diff = np.array(seed - mutate_sample).flatten()
        size = np.shape(diff)[0]
        l0 = np.linalg.norm(diff, ord=0)
        linf = np.linalg.norm(diff, ord=np.inf)
        if l0 > pixels_change_rate*size:
            if linf < 256:
                is_valid = True
        else:
            if linf < pixel_value_change_rate*255:
                is_valid = True
        return is_valid

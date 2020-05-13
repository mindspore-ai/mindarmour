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
    check_int_positive
from mindarmour.utils.image_transform import Contrast, Brightness, Blur, Noise, \
    Translate, Scale, Shear, Rotate


class Fuzzing:
    """
    Fuzzing test framework for deep neural networks.

    Reference: `DeepHunter: A Coverage-Guided Fuzz Testing Framework for Deep
    Neural Networks <https://dl.acm.org/doi/10.1145/3293882.3330579>`_

    Args:
        initial_seeds (list): Initial fuzzing seed, format: [[image, label, 0],
            [image, label, 0], ...].
        target_model (Model): Target fuzz model.
        train_dataset (numpy.ndarray): Training dataset used for determine
            the neurons' output boundaries.
        const_k (int): The number of mutate tests for a seed.
        mode (str): Image mode used in image transform, 'L' means grey graph.
            Default: 'L'.
        max_seed_num (int): The initial seeds max value. Default: 1000
    """

    def __init__(self, initial_seeds, target_model, train_dataset, const_K,
                 mode='L', max_seed_num=1000):
        self.initial_seeds = initial_seeds
        self.target_model = check_model('model', target_model, Model)
        self.train_dataset = check_numpy_param('train_dataset', train_dataset)
        self.const_k = check_int_positive('const_k', const_K)
        self.mode = mode
        self.max_seed_num = check_int_positive('max_seed_num', max_seed_num)
        self.coverage_metrics = ModelCoverageMetrics(target_model, 1000, 10,
                                                     train_dataset)

    def _image_value_expand(self, image):
        return image*255

    def _image_value_compress(self, image):
        return image / 255

    def _metamorphic_mutate(self, seed, try_num=50):
        if self.mode == 'L':
            seed = seed[0]
        info = [seed, seed]
        mutate_tests = []
        affine_trans = ['Contrast', 'Brightness', 'Blur', 'Noise']
        pixel_value_trans = ['Translate', 'Scale', 'Shear', 'Rotate']
        strages = {'Contrast': Contrast, 'Brightness': Brightness, 'Blur': Blur,
                   'Noise': Noise,
                   'Translate': Translate, 'Scale': Scale, 'Shear': Shear,
                   'Rotate': Rotate}
        for _ in range(self.const_k):
            for _ in range(try_num):
                if (info[0] == info[1]).all():
                    trans_strage = self._random_pick_mutate(affine_trans,
                                                            pixel_value_trans)
                else:
                    trans_strage = self._random_pick_mutate(affine_trans, [])
                transform = strages[trans_strage](
                    self._image_value_expand(seed), self.mode)
                transform.random_param()
                mutate_test = transform.transform()
                mutate_test = np.expand_dims(
                    self._image_value_compress(mutate_test), 0)

                if self._is_trans_valid(seed, mutate_test):
                    if trans_strage in affine_trans:
                        info[1] = mutate_test
                    mutate_tests.append(mutate_test)
            if not mutate_tests:
                mutate_tests.append(seed)
            return np.array(mutate_tests)

    def fuzzing(self, coverage_metric='KMNC'):
        """
        Fuzzing tests for deep neural networks.

        Args:
            coverage_metric (str): Model coverage metric of neural networks.
                Default: 'KMNC'.

        Returns:
            list, mutated tests mis-predicted by target dnn model.
        """
        seed = self._select_next()
        failed_tests = []
        seed_num = 0
        while seed and seed_num < self.max_seed_num:
            mutate_tests = self._metamorphic_mutate(seed[0])
            coverages, results = self._run(mutate_tests, coverage_metric)
            coverage_gains = self._coverage_gains(coverages)
            for mutate, cov, res in zip(mutate_tests, coverage_gains, results):
                if np.argmax(seed[1]) != np.argmax(res):
                    failed_tests.append(mutate)
                    continue
                if cov > 0:
                    self.initial_seeds.append([mutate, seed[1], 0])
            seed = self._select_next()
            seed_num += 1

        return failed_tests

    def _coverage_gains(self, coverages):
        gains = [0] + coverages[:-1]
        gains = np.array(coverages) - np.array(gains)
        return gains

    def _run(self, mutate_tests, coverage_metric="KNMC"):
        coverages = []
        result = self.target_model.predict(
            Tensor(mutate_tests.astype(np.float32)))
        result = result.asnumpy()
        for index in range(len(mutate_tests)):
            mutate = np.expand_dims(mutate_tests[index], 0)
            self.coverage_metrics.test_adequacy_coverage_calculate(
                mutate.astype(np.float32), batch_size=1)
            if coverage_metric == "KMNC":
                coverages.append(self.coverage_metrics.get_kmnc())

        return coverages, result

    def _select_next(self):
        seed = choice(self.initial_seeds)
        return seed

    def _random_pick_mutate(self, affine_trans_list, pixel_value_trans_list):
        strage = choice(affine_trans_list + pixel_value_trans_list)
        return strage

    def _is_trans_valid(self, seed, mutate_test):
        is_valid = False
        alpha = 0.02
        beta = 0.2
        diff = np.array(seed - mutate_test).flatten()
        size = np.shape(diff)[0]
        l0 = np.linalg.norm(diff, ord=0)
        linf = np.linalg.norm(diff, ord=np.inf)
        if l0 > alpha*size:
            if linf < 256:
                is_valid = True
        else:
            if linf < beta*255:
                is_valid = True

        return is_valid

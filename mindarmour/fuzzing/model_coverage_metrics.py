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
Model-Test Coverage Metrics.
"""

import numpy as np

from mindspore import Tensor
from mindspore import Model

from mindarmour.utils._check_param import check_model, check_numpy_param, \
    check_int_positive


class ModelCoverageMetrics:
    """
    Evaluate the testing adequacy of a model fuzz test.

    Reference: `DeepGauge: Multi-Granularity Testing Criteria for Deep
    Learning Systems <https://arxiv.org/abs/1803.07519>`_

    Args:
        model (Model): The pre-trained model which waiting for testing.
        k (int): The number of segmented sections of neurons' output intervals.
        n (int): The number of testing neurons.
        train_dataset (numpy.ndarray): Training dataset used for determine
            the neurons' output boundaries.

    Examples:
        >>> train_images = np.random.random((10000, 128)).astype(np.float32)
        >>> test_images = np.random.random((5000, 128)).astype(np.float32)
        >>> model = Model(net)
        >>> model_fuzz_test = ModelCoverageMetrics(model, 10000, 10, train_images)
        >>> model_fuzz_test.test_adequacy_coverage_calculate(test_images)
        >>> print('KMNC of this test is : %s', model_fuzz_test.get_kmnc())
        >>> print('NBC of this test is : %s', model_fuzz_test.get_nbc())
        >>> print('SNAC of this test is : %s', model_fuzz_test.get_snac())
    """

    def __init__(self, model, k, n, train_dataset):
        self._model = check_model('model', model, Model)
        self._k = k
        self._n = n
        train_dataset = check_numpy_param('train_dataset', train_dataset)
        self._lower_bounds = [np.inf]*n
        self._upper_bounds = [-np.inf]*n
        self._var = [0]*n
        self._main_section_hits = [[0 for _ in range(self._k)] for _ in
                                   range(self._n)]
        self._lower_corner_hits = [0]*self._n
        self._upper_corner_hits = [0]*self._n
        self._bounds_get(train_dataset)

    def _bounds_get(self, train_dataset, batch_size=32):
        """
        Update the lower and upper boundaries of neurons' outputs.

        Args:
            train_dataset (numpy.ndarray): Training dataset used for
                determine the neurons' output boundaries.
            batch_size (int): The number of samples in a predict batch.
                Default: 32.
        """
        batch_size = check_int_positive('batch_size', batch_size)
        output_mat = []
        batches = train_dataset.shape[0] // batch_size
        for i in range(batches):
            inputs = train_dataset[i*batch_size: (i + 1)*batch_size]
            output = self._model.predict(Tensor(inputs)).asnumpy()
            output_mat.append(output)
            lower_compare_array = np.concatenate(
                [output, np.array([self._lower_bounds])], axis=0)
            self._lower_bounds = np.min(lower_compare_array, axis=0)
            upper_compare_array = np.concatenate(
                [output, np.array([self._upper_bounds])], axis=0)
            self._upper_bounds = np.max(upper_compare_array, axis=0)
        if batches == 0:
            output = self._model.predict(Tensor(train_dataset)).asnumpy()
            self._lower_bounds = np.min(output, axis=0)
            self._upper_bounds = np.max(output, axis=0)
            output_mat.append(output)
        self._var = np.std(np.concatenate(np.array(output_mat), axis=0),
                           axis=0)

    def _sections_hits_count(self, dataset, intervals):
        """
        Update the coverage matrix of neurons' output subsections.

        Args:
            dataset (numpy.ndarray): Testing data.
            intervals (list[float]): Segmentation intervals of neurons'
                outputs.
        """
        dataset = check_numpy_param('dataset', dataset)
        batch_output = self._model.predict(Tensor(dataset)).asnumpy()
        batch_section_indexes = (batch_output - self._lower_bounds) // intervals
        for section_indexes in batch_section_indexes:
            for i in range(self._n):
                if section_indexes[i] < 0:
                    self._lower_corner_hits[i] = 1
                elif section_indexes[i] >= self._k:
                    self._upper_corner_hits[i] = 1
                else:
                    self._main_section_hits[i][int(section_indexes[i])] = 1

    def test_adequacy_coverage_calculate(self, dataset, bias_coefficient=0,
                                         batch_size=32):
        """
        Calculate the testing adequacy of the given dataset.

        Args:
            dataset (numpy.ndarray): Data for fuzz test.
            bias_coefficient (float): The coefficient used for changing the
                neurons' output boundaries. Default: 0.
            batch_size (int): The number of samples in a predict batch.
                Default: 32.

        Examples:
            >>> model_fuzz_test = ModelCoverageMetrics(model, 10000, 10, train_images)
            >>> model_fuzz_test.test_adequacy_coverage_calculate(test_images)
        """
        dataset = check_numpy_param('dataset', dataset)
        batch_size = check_int_positive('batch_size', batch_size)
        self._lower_bounds -= bias_coefficient*self._var
        self._upper_bounds += bias_coefficient*self._var
        intervals = (self._upper_bounds - self._lower_bounds) / self._k
        batches = dataset.shape[0] // batch_size
        for i in range(batches):
            self._sections_hits_count(
                dataset[i*batch_size: (i + 1)*batch_size], intervals)

    def get_kmnc(self):
        """
        Get the metric of 'k-multisection neuron coverage'.

        Returns:
            float, the metric of 'k-multisection neuron coverage'.

        Examples:
            >>> model_fuzz_test.get_kmnc()
        """
        kmnc = np.sum(self._main_section_hits) / (self._n*self._k)
        return kmnc

    def get_nbc(self):
        """
        Get the metric of 'neuron boundary coverage'.

        Returns:
            float, the metric of 'neuron boundary coverage'.

        Examples:
            >>> model_fuzz_test.get_nbc()
        """
        nbc = (np.sum(self._lower_corner_hits) + np.sum(
            self._upper_corner_hits)) / (2*self._n)
        return nbc

    def get_snac(self):
        """
        Get the metric of 'strong neuron activation coverage'.

        Returns:
            float, the metric of 'strong neuron activation coverage'.

        Examples:
            >>> model_fuzz_test.get_snac()
        """
        snac = np.sum(self._upper_corner_hits) / self._n
        return snac

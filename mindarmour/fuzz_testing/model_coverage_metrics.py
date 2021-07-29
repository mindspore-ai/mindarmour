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

from collections import defaultdict
import numpy as np

from mindspore import Tensor
from mindspore import Model
from mindspore.train.summary.summary_record import _get_summary_tensor_data

from mindarmour.utils._check_param import check_model, check_numpy_param, \
    check_int_positive, check_param_multi_types
from mindarmour.utils.logger import LogUtil

LOGGER = LogUtil.get_instance()
TAG = 'ModelCoverageMetrics'


class ModelCoverageMetrics:
    """
    As we all known, each neuron output of a network will have a output range
    after training (we call it original range), and test dataset is used to
    estimate the accuracy of the trained network. However, neurons' output
    distribution would be different with different test datasets. Therefore,
    similar to function fuzz, model fuzz means testing those neurons' outputs
    and estimating the proportion of original range that has emerged with test
    datasets.

    Reference: `DeepGauge: Multi-Granularity Testing Criteria for Deep
    Learning Systems <https://arxiv.org/abs/1803.07519>`_

    Args:
        model (Model): The pre-trained model which waiting for testing.
        neuron_num (int): The number of testing neurons.
        segmented_num (int): The number of segmented sections of neurons' output intervals.
        train_dataset (numpy.ndarray): Training dataset used for determine
            the neurons' output boundaries.

    Raises:
        ValueError: If neuron_num is too big (for example, bigger than 1e+9).

    Examples:
        >>> net = LeNet5()
        >>> train_images = np.random.random((10000, 1, 32, 32)).astype(np.float32)
        >>> test_images = np.random.random((5000, 1, 32, 32)).astype(np.float32)
        >>> model = Model(net)
        >>> neuron_num = 10
        >>> segmented_num = 1000
        >>> model_fuzz_test = ModelCoverageMetrics(model, neuron_num, segmented_num, train_images)
        >>> model_fuzz_test.calculate_coverage(test_images)
        >>> print('KMNC of this test is : %s', model_fuzz_test.get_kmnc())
        >>> print('NBC of this test is : %s', model_fuzz_test.get_nbc())
        >>> print('SNAC of this test is : %s', model_fuzz_test.get_snac())
        >>> model_fuzz_test.calculate_effective_coverage(test_images, top_k, threshold)
        >>> print('NC of this test is : %s', model_fuzz_test.get_nc())
        >>> print('Effective_NC of this test is : %s', model_fuzz_test.get_effective_nc())
    """

    def __init__(self, model, neuron_num, segmented_num, train_dataset):
        self._model = check_model('model', model, Model)
        self._segmented_num = check_int_positive('segmented_num', segmented_num)
        self._neuron_num = check_int_positive('neuron_num', neuron_num)
        if self._neuron_num > 1e+9:
            msg = 'neuron_num should be less than 1e+10, otherwise a MemoryError would occur'
            LOGGER.error(TAG, msg)
            raise ValueError(msg)
        train_dataset = check_numpy_param('train_dataset', train_dataset)
        self._lower_bounds = [np.inf]*self._neuron_num
        self._upper_bounds = [-np.inf]*self._neuron_num
        self._var = [0]*self._neuron_num
        self._main_section_hits = [[0 for _ in range(self._segmented_num)] for _ in range(self._neuron_num)]
        self._lower_corner_hits = [0]*self._neuron_num
        self._upper_corner_hits = [0]*self._neuron_num
        self._bounds_get(train_dataset)
        self._model_layer_dict = defaultdict(bool)
        self._effective_model_layer_dict = defaultdict(bool)

    def _set_init_effective_coverage_table(self, dataset):
        """
        Initialise the coverage table of each neuron in the model.

        Args:
            dataset (numpy.ndarray): Dataset used for initialising the coverage table.
        """
        self._model.predict(Tensor(dataset[0:1]))
        tensors = _get_summary_tensor_data()
        for name, tensor in tensors.items():
            if 'input' in name:
                continue
            for num_neuron in range(tensor.shape[1]):
                self._model_layer_dict[(name, num_neuron)] = False
                self._effective_model_layer_dict[(name, num_neuron)] = False

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
            lower_compare_array = np.concatenate([output, np.array([self._lower_bounds])], axis=0)
            self._lower_bounds = np.min(lower_compare_array, axis=0)
            upper_compare_array = np.concatenate([output, np.array([self._upper_bounds])], axis=0)
            self._upper_bounds = np.max(upper_compare_array, axis=0)
        if batches == 0:
            output = self._model.predict(Tensor(train_dataset)).asnumpy()
            self._lower_bounds = np.min(output, axis=0)
            self._upper_bounds = np.max(output, axis=0)
            output_mat.append(output)
        self._var = np.std(np.concatenate(np.array(output_mat), axis=0), axis=0)

    def _sections_hits_count(self, dataset, intervals):
        """
        Update the coverage matrix of neurons' output subsections.

        Args:
            dataset (numpy.ndarray): Testing data.
            intervals (list[float]): Segmentation intervals of neurons' outputs.
        """
        dataset = check_numpy_param('dataset', dataset)
        batch_output = self._model.predict(Tensor(dataset)).asnumpy()
        batch_section_indexes = (batch_output - self._lower_bounds) // intervals
        for section_indexes in batch_section_indexes:
            for i in range(self._neuron_num):
                if section_indexes[i] < 0:
                    self._lower_corner_hits[i] = 1
                elif section_indexes[i] >= self._segmented_num:
                    self._upper_corner_hits[i] = 1
                else:
                    self._main_section_hits[i][int(section_indexes[i])] = 1

    def _coverage_update(self, name, tensor, scaled_mean, scaled_rank, top_k, threshold):
        """
        Update the coverage matrix of neural coverage and effective neural coverage.

        Args:
            name (string): the name of the tensor.
            tensor (tensor): the tensor in the network.
            scaled_mean (numpy.ndarray): feature map of the tensor.
            scaled_rank (numpy.ndarray): rank of tensor value.
            top_k (int): neuron is covered when its output has the top k largest value in that hidden layer.
            threshold (float): neuron is covered when its output is greater than the threshold.

        """
        for num_neuron in range(tensor.shape[1]):
            if num_neuron >= (len(scaled_rank) - top_k) and not \
                    self._effective_model_layer_dict[(name, scaled_rank[num_neuron])]:
                self._effective_model_layer_dict[(name, scaled_rank[num_neuron])] = True
            if scaled_mean[num_neuron] > threshold and not \
                    self._model_layer_dict[(name, num_neuron)]:
                self._model_layer_dict[(name, num_neuron)] = True

    def calculate_coverage(self, dataset, bias_coefficient=0, batch_size=32):
        """
        Calculate the testing adequacy of the given dataset.

        Args:
            dataset (numpy.ndarray): Data for fuzz test.
            bias_coefficient (Union[int, float]): The coefficient used
                for changing the neurons' output boundaries. Default: 0.
            batch_size (int): The number of samples in a predict batch. Default: 32.

        Examples:
            >>> neuron_num = 10
            >>> segmented_num = 1000
            >>> batch_size = 32
            >>> model_fuzz_test = ModelCoverageMetrics(model, neuron_num, segmented_num, train_images)
            >>> model_fuzz_test.calculate_coverage(test_images, top_k, threshold, batch_size)
        """

        dataset = check_numpy_param('dataset', dataset)
        batch_size = check_int_positive('batch_size', batch_size)
        bias_coefficient = check_param_multi_types('bias_coefficient', bias_coefficient, [int, float])
        self._lower_bounds -= bias_coefficient*self._var
        self._upper_bounds += bias_coefficient*self._var
        intervals = (self._upper_bounds - self._lower_bounds) / self._segmented_num
        batches = dataset.shape[0] // batch_size
        for i in range(batches):
            self._sections_hits_count(dataset[i*batch_size: (i + 1)*batch_size], intervals)


    def calculate_effective_coverage(self, dataset, top_k=3, threshold=0.1, batch_size=32):
        """
        Calculate the effective testing adequacy of the given dataset.
        In effective neural coverage, neuron is covered when its output has the top k largest value
        in that hidden layers. In neural coverage, neuron is covered when its output is greater than the
        threshold. Coverage equals the covered neurons divided by the total neurons in the network.

        Args:
            threshold (float): neuron is covered when its output is greater than the threshold.
            top_k (int): neuron is covered when its output has the top k largest value in that hiddern layer.
            dataset (numpy.ndarray): Data for fuzz test.

        Examples:
            >>> neuron_num = 10
            >>> segmented_num = 1000
            >>> top_k = 3
            >>> threshold = 0.1
            >>> model_fuzz_test = ModelCoverageMetrics(model, neuron_num, segmented_num, train_images)
            >>> model_fuzz_test.calculate_coverage(test_images)
            >>> model_fuzz_test.calculate_effective_coverage(test_images, top_k, threshold)
        """
        top_k = check_int_positive('top_k', top_k)
        dataset = check_numpy_param('dataset', dataset)
        batch_size = check_int_positive('batch_size', batch_size)
        batches = dataset.shape[0] // batch_size
        self._set_init_effective_coverage_table(dataset)
        for i in range(batches):
            inputs = dataset[i*batch_size: (i + 1)*batch_size]
            self._model.predict(Tensor(inputs)).asnumpy()
            tensors = _get_summary_tensor_data()
            for name, tensor in tensors.items():
                if 'input' in name:
                    continue
                scaled = tensor.asnumpy()[-1]
                if scaled.ndim >= 3:  #
                    scaled_mean = np.mean(scaled, axis=(1, 2))
                    scaled_rank = np.argsort(scaled_mean)
                    self._coverage_update(name, tensor, scaled_mean, scaled_rank, top_k, threshold)
                else:
                    scaled_rank = np.argsort(scaled)
                    self._coverage_update(name, tensor, scaled, scaled_rank, top_k, threshold)

    def get_nc(self):
        """
        Get the metric of 'neuron coverage'.

        Returns:
            float, the metric of 'neuron coverage'.

        Examples:
            >>> model_fuzz_test.get_nc()
        """
        covered_neurons = len([v for v in self._model_layer_dict.values() if v])
        total_neurons = len(self._model_layer_dict)
        nc = covered_neurons / float(total_neurons)
        return nc

    def get_effective_nc(self):
        """
        Get the metric of 'effective neuron coverage'.

        Returns:
            float, the metric of 'the effective neuron coverage'.

        Examples:
            >>> model_fuzz_test.get_effective_nc()
        """
        covered_neurons = len([v for v in self._effective_model_layer_dict.values() if v])
        total_neurons = len(self._effective_model_layer_dict)
        effective_nc = covered_neurons / float(total_neurons)
        return effective_nc

    def get_kmnc(self):
        """
        Get the metric of 'k-multisection neuron coverage'. KMNC measures how
        thoroughly the given set of test inputs covers the range of neurons
        output values derived from training dataset.

        Returns:
            float, the metric of 'k-multisection neuron coverage'.

        Examples:
            >>> model_fuzz_test.get_kmnc()
        """
        kmnc = np.sum(self._main_section_hits) / (self._neuron_num*self._segmented_num)
        return kmnc

    def get_nbc(self):
        """
        Get the metric of 'neuron boundary coverage' :math:`NBC = (|UpperCornerNeuron|
        + |LowerCornerNeuron|)/(2*|N|)`, where :math`|N|` is the number of neurons,
        NBC refers to the proportion of neurons whose neurons output value in
        the test dataset exceeds the upper and lower bounds of the corresponding
        neurons output value in the training dataset.

        Returns:
            float, the metric of 'neuron boundary coverage'.

        Examples:
            >>> model_fuzz_test.get_nbc()
        """
        nbc = (np.sum(self._lower_corner_hits) + np.sum(self._upper_corner_hits)) / (2*self._neuron_num)
        return nbc

    def get_snac(self):
        """
        Get the metric of 'strong neuron activation coverage'.
        :math:`SNAC = |UpperCornerNeuron|/|N|`. SNAC refers to the proportion
        of neurons whose neurons output value in the test set exceeds the upper
        bounds of the corresponding neurons output value in the training set.

        Returns:
            float, the metric of 'strong neuron activation coverage'.

        Examples:
            >>> model_fuzz_test.get_snac()
        """
        snac = np.sum(self._upper_corner_hits) / self._neuron_num
        return snac

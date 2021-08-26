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
from abc import abstractmethod
from collections import defaultdict
import math
import numpy as np

from mindspore import Tensor
from mindspore import Model
from mindspore.train.summary.summary_record import _get_summary_tensor_data

from mindarmour.utils._check_param import check_model, check_numpy_param, check_int_positive, \
    check_param_type, check_value_positive
from mindarmour.utils.logger import LogUtil

LOGGER = LogUtil.get_instance()
TAG = 'CoverageMetrics'


class CoverageMetrics:
    """
    The abstract base class for Neuron coverage classes calculating coverage metrics.

    As we all known, each neuron output of a network will have a output range after training (we call it original
    range), and test dataset is used to estimate the accuracy of the trained network. However, neurons' output
    distribution would be different with different test datasets. Therefore, similar to function fuzz, model fuzz means
    testing those neurons' outputs and estimating the proportion of original range that has emerged with test
    datasets.

    Reference: `DeepGauge: Multi-Granularity Testing Criteria for Deep Learning Systems
    <https://arxiv.org/abs/1803.07519>`_

    Args:
        model (Model): The pre-trained model which waiting for testing.
        incremental (bool): Metrics will be calculate in incremental way or not. Default: False.
        batch_size (int):  The number of samples in a fuzz test batch. Default: 32.
    """

    def __init__(self, model, incremental=False, batch_size=32):
        self._model = check_model('model', model, Model)
        self.incremental = check_param_type('incremental', incremental, bool)
        self.batch_size = check_int_positive('batch_size', batch_size)
        self._activate_table = defaultdict(list)

    @abstractmethod
    def get_metrics(self, dataset):
        """
        Calculate coverage metrics of given dataset.

        Args:
            dataset (numpy.ndarray): Dataset used to calculate coverage metrics.

        Raises:
            NotImplementedError: It is an abstract method.
        """
        msg = 'The function get_metrics() is an abstract method in class `CoverageMetrics`, and should be' \
              ' implemented in child class.'
        LOGGER.error(TAG, msg)
        raise NotImplementedError(msg)

    def _init_neuron_activate_table(self, data):
        """
        Initialise the activate table of each neuron in the model with format:
        {'layer1': [n1, n2, n3, ..., nn], 'layer2': [n1, n2, n3, ..., nn], ...}

        Args:
            data (numpy.ndarray): Data used for initialising the activate table.

        Return:
            dict, return a activate_table.
        """
        self._model.predict(Tensor(data))
        layer_out = _get_summary_tensor_data()
        if not layer_out:
            msg = 'User must use TensorSummary() operation to specify the middle layer of the model participating in ' \
                  'the coverage calculation.'
            LOGGER.error(TAG, msg)
            raise ValueError(msg)
        activate_table = defaultdict()
        for layer, value in layer_out.items():
            activate_table[layer] = np.zeros(value.shape[1], np.bool)
        return activate_table

    def _get_bounds(self, train_dataset):
        """
        Update the lower and upper boundaries of neurons' outputs.

        Args:
            train_dataset (numpy.ndarray): Training dataset used for determine the neurons' output boundaries.

        Return:
            - numpy.ndarray, upper bounds of neuron' outputs.

            - numpy.ndarray, lower bounds of neuron' outputs.
        """
        upper_bounds = defaultdict(list)
        lower_bounds = defaultdict(list)
        batches = math.ceil(train_dataset.shape[0] / self.batch_size)
        for i in range(batches):
            inputs = train_dataset[i * self.batch_size: (i + 1) * self.batch_size]
            self._model.predict(Tensor(inputs))
            layer_out = _get_summary_tensor_data()
            for layer, tensor in layer_out.items():
                value = tensor.asnumpy()
                value = np.mean(value, axis=tuple([i for i in range(2, len(value.shape))]))
                min_value = np.min(value, axis=0)
                max_value = np.max(value, axis=0)
                if np.any(upper_bounds[layer]):
                    max_flag = upper_bounds[layer] > max_value
                    min_flag = lower_bounds[layer] < min_value
                    upper_bounds[layer] = upper_bounds[layer] * max_flag + max_value * (1 - max_flag)
                    lower_bounds[layer] = lower_bounds[layer] * min_flag + min_value * (1 - min_flag)
                else:
                    upper_bounds[layer] = max_value
                    lower_bounds[layer] = min_value
        return upper_bounds, lower_bounds

    def _activate_rate(self):
        """
        Calculate the activate rate of neurons.
        """
        total_neurons = 0
        activated_neurons = 0
        for _, value in self._activate_table.items():
            activated_neurons += np.sum(value)
            total_neurons += len(value)
        activate_rate = activated_neurons / total_neurons

        return activate_rate


class NeuronCoverage(CoverageMetrics):
    """
    Calculate the neurons activated coverage. Neuron is activated when its output is greater than the threshold.
    Neuron coverage equals the proportion of activated neurons to total neurons in the network.

    Args:
        model (Model): The pre-trained model which waiting for testing.
        threshold (float): Threshold used to determined neurons is activated or not. Default: 0.1.
        incremental (bool): Metrics will be calculate in incremental way or not. Default: False.
        batch_size (int):  The number of samples in a fuzz test batch. Default: 32.

    """
    def __init__(self, model, threshold=0.1, incremental=False, batch_size=32):
        super(NeuronCoverage, self).__init__(model, incremental, batch_size)
        threshold = check_param_type('threshold', threshold, float)
        self.threshold = check_value_positive('threshold', threshold)


    def get_metrics(self, dataset):
        """
        Get the metric of neuron coverage: the proportion of activated neurons to total neurons in the network.

        Args:
            dataset (numpy.ndarray): Dataset used to calculate coverage metrics.

        Returns:
            float, the metric of 'neuron coverage'.

        Examples:
            >>> nc = NeuronCoverage(model, threshold=0.1)
            >>> nc_metrics = nc.get_metrics(test_data)
        """
        dataset = check_numpy_param('dataset', dataset)
        batches = math.ceil(dataset.shape[0] / self.batch_size)
        if not self.incremental or not self._activate_table:
            self._activate_table = self._init_neuron_activate_table(dataset[0:1])
        for i in range(batches):
            inputs = dataset[i * self.batch_size: (i + 1) * self.batch_size]
            self._model.predict(Tensor(inputs))
            layer_out = _get_summary_tensor_data()
            for layer, tensor in layer_out.items():
                value = tensor.asnumpy()
                value = np.mean(value, axis=tuple([i for i in range(2, len(value.shape))]))
                activate = np.sum(value > self.threshold, axis=0) > 0
                self._activate_table[layer] = np.logical_or(self._activate_table[layer], activate)
        neuron_coverage = self._activate_rate()
        return neuron_coverage


class TopKNeuronCoverage(CoverageMetrics):
    """
    Calculate the top k activated neurons coverage. Neuron is activated when its output has the top k largest value in
    that hidden layers. Top k neurons coverage equals the proportion of activated neurons to total neurons in the
    network.

    Args:
        model (Model): The pre-trained model which waiting for testing.
        top_k (int): Neuron is activated when its output has the top k largest value in that hidden layers. Default: 3.
        incremental (bool): Metrics will be calculate in incremental way or not. Default: False.
        batch_size (int):  The number of samples in a fuzz test batch. Default: 32.
    """
    def __init__(self, model, top_k=3, incremental=False, batch_size=32):
        super(TopKNeuronCoverage, self).__init__(model, incremental=incremental, batch_size=batch_size)
        self.top_k = check_int_positive('top_k', top_k)

    def get_metrics(self, dataset):
        """
        Get the metric of Top K activated neuron coverage.

        Args:
            dataset (numpy.ndarray): Dataset used to calculate coverage metrics.

        Returns:
            float, the metrics of 'top k neuron coverage'.

        Examples:
            >>> tknc = TopKNeuronCoverage(model, top_k=3)
            >>> metrics = tknc.get_metrics(test_data)
        """
        dataset = check_numpy_param('dataset', dataset)
        batches = math.ceil(dataset.shape[0] / self.batch_size)
        if not self.incremental or not self._activate_table:
            self._activate_table = self._init_neuron_activate_table(dataset[0:1])
        for i in range(batches):
            inputs = dataset[i * self.batch_size: (i + 1) * self.batch_size]
            self._model.predict(Tensor(inputs))
            layer_out = _get_summary_tensor_data()
            for layer, tensor in layer_out.items():
                value = tensor.asnumpy()
                if len(value.shape) > 2:
                    value = np.mean(value, axis=tuple([i for i in range(2, len(value.shape))]))
                top_k_value = np.sort(value)[:, -self.top_k].reshape(value.shape[0], 1)
                top_k_value = np.sum((value - top_k_value) >= 0, axis=0) > 0
                self._activate_table[layer] = np.logical_or(self._activate_table[layer], top_k_value)
        top_k_neuron_coverage = self._activate_rate()
        return top_k_neuron_coverage


class SuperNeuronActivateCoverage(CoverageMetrics):
    """
    Get the metric of 'super neuron activation coverage'. :math:`SNAC = |UpperCornerNeuron|/|N|`. SNAC refers to the
    proportion of neurons whose neurons output value in the test set exceeds the upper bounds of the corresponding
    neurons output value in the training set.

    Args:
        model (Model): The pre-trained model which waiting for testing.
        train_dataset (numpy.ndarray): Training dataset used for determine the neurons' output boundaries.
        incremental (bool): Metrics will be calculate in incremental way or not. Default: False.
        batch_size (int):  The number of samples in a fuzz test batch. Default: 32.
    """
    def __init__(self, model, train_dataset, incremental=False, batch_size=32):
        super(SuperNeuronActivateCoverage, self).__init__(model, incremental=incremental, batch_size=batch_size)
        train_dataset = check_numpy_param('train_dataset', train_dataset)
        self.upper_bounds, self.lower_bounds = self._get_bounds(train_dataset=train_dataset)

    def get_metrics(self, dataset):
        """
        Get the metric of 'strong neuron activation coverage'.

        Args:
            dataset (numpy.ndarray): Dataset used to calculate coverage metrics.

        Returns:
            float, the metric of 'strong neuron activation coverage'.

        Examples:
            >>> snac = SuperNeuronActivateCoverage(model, train_dataset)
            >>> metrics = snac.get_metrics(test_data)
        """
        dataset = check_numpy_param('dataset', dataset)
        if not self.incremental or not self._activate_table:
            self._activate_table = self._init_neuron_activate_table(dataset[0:1])
        batches = math.ceil(dataset.shape[0] / self.batch_size)

        for i in range(batches):
            inputs = dataset[i * self.batch_size: (i + 1) * self.batch_size]
            self._model.predict(Tensor(inputs))
            layer_out = _get_summary_tensor_data()
            for layer, tensor in layer_out.items():
                value = tensor.asnumpy()
                if len(value.shape) > 2:
                    value = np.mean(value, axis=tuple([i for i in range(2, len(value.shape))]))
                activate = np.sum(value > self.upper_bounds[layer], axis=0) > 0
                self._activate_table[layer] = np.logical_or(self._activate_table[layer], activate)
        snac = self._activate_rate()
        return snac


class NeuronBoundsCoverage(SuperNeuronActivateCoverage):
    """
    Get the metric of 'neuron boundary coverage' :math:`NBC = (|UpperCornerNeuron| + |LowerCornerNeuron|)/(2*|N|)`,
    where :math`|N|` is the number of neurons, NBC refers to the proportion of neurons whose neurons output value in
    the test dataset exceeds the upper and lower bounds of the corresponding neurons output value in the training
    dataset.

    Args:
        model (Model): The pre-trained model which waiting for testing.
        train_dataset (numpy.ndarray): Training dataset used for determine the neurons' output boundaries.
        incremental (bool): Metrics will be calculate in incremental way or not. Default: False.
        batch_size (int):  The number of samples in a fuzz test batch. Default: 32.
    """

    def __init__(self, model, train_dataset, incremental=False, batch_size=32):
        super(NeuronBoundsCoverage, self).__init__(model, train_dataset, incremental=incremental, batch_size=batch_size)

    def get_metrics(self, dataset):
        """
        Get the metric of 'neuron boundary coverage'.

        Args:
            dataset (numpy.ndarray): Dataset used to calculate coverage metrics.

        Returns:
            float, the metric of 'neuron boundary coverage'.

        Examples:
            >>> nbc = NeuronBoundsCoverage(model, train_dataset)
            >>> metrics = nbc.get_metrics(test_data)
        """
        dataset = check_numpy_param('dataset', dataset)
        if not self.incremental or not self._activate_table:
            self._activate_table = self._init_neuron_activate_table(dataset[0:1])

        batches = math.ceil(dataset.shape[0] / self.batch_size)
        for i in range(batches):
            inputs = dataset[i * self.batch_size: (i + 1) * self.batch_size]
            self._model.predict(Tensor(inputs))
            layer_out = _get_summary_tensor_data()
            for layer, tensor in layer_out.items():
                value = tensor.asnumpy()
                if len(value.shape) > 2:
                    value = np.mean(value, axis=tuple([i for i in range(2, len(value.shape))]))
                outer = np.logical_or(value > self.upper_bounds[layer], value < self.lower_bounds[layer])
                activate = np.sum(outer, axis=0) > 0
                self._activate_table[layer] = np.logical_or(self._activate_table[layer], activate)
        nbc = self._activate_rate()
        return nbc


class KMultisectionNeuronCoverage(SuperNeuronActivateCoverage):
    """
    Get the metric of 'k-multisection neuron coverage'. KMNC measures how thoroughly the given set of test inputs
    covers the range of neurons output values derived from training dataset.

    Args:
        model (Model): The pre-trained model which waiting for testing.
        train_dataset (numpy.ndarray): Training dataset used for determine the neurons' output boundaries.
        segmented_num (int): The number of segmented sections of neurons' output intervals. Default: 100.
        incremental (bool): Metrics will be calculate in incremental way or not. Default: False.
        batch_size (int):  The number of samples in a fuzz test batch. Default: 32.
    """

    def __init__(self, model, train_dataset, segmented_num=100, incremental=False, batch_size=32):
        super(KMultisectionNeuronCoverage, self).__init__(model, train_dataset, incremental=incremental,
                                                          batch_size=batch_size)
        self.segmented_num = check_int_positive('segmented_num', segmented_num)
        self.intervals = defaultdict(list)
        for keys in self.upper_bounds.keys():
            self.intervals[keys] = (self.upper_bounds[keys] - self.lower_bounds[keys]) / self.segmented_num

    def _init_k_multisection_table(self, data):
        """ Initial the activate table."""
        self._model.predict(Tensor(data))
        layer_out = _get_summary_tensor_data()
        activate_section_table = defaultdict()
        for layer, value in layer_out.items():
            activate_section_table[layer] = np.zeros((value.shape[1], self.segmented_num), np.bool)
        return activate_section_table

    def get_metrics(self, dataset):
        """
        Get the metric of 'k-multisection neuron coverage'.

        Args:
            dataset (numpy.ndarray): Dataset used to calculate coverage metrics.

        Returns:
            float, the metric of 'k-multisection neuron coverage'.

        Examples:
            >>> kmnc = KMultisectionNeuronCoverage(model, train_dataset, segmented_num=100)
            >>> metrics = kmnc.get_metrics(test_data)
        """

        dataset = check_numpy_param('dataset', dataset)
        if not self.incremental or not self._activate_table:
            self._activate_table = self._init_k_multisection_table(dataset[0:1])

        batches = math.ceil(dataset.shape[0] / self.batch_size)
        for i in range(batches):
            inputs = dataset[i * self.batch_size: (i + 1) * self.batch_size]
            self._model.predict(Tensor(inputs))
            layer_out = _get_summary_tensor_data()
            for layer, tensor in layer_out.items():
                value = tensor.asnumpy()
                value = np.mean(value, axis=tuple([i for i in range(2, len(value.shape))]))
                hits = np.floor((value - self.lower_bounds[layer]) / self.intervals[layer]).astype(int)
                hits = np.transpose(hits, [1, 0])
                for n in range(len(hits)):
                    for sec in hits[n]:
                        if sec >= self.segmented_num or sec < 0:
                            continue
                        self._activate_table[layer][n][sec] = True

        kmnc = self._activate_rate() / self.segmented_num
        return kmnc

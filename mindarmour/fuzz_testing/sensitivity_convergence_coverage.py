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
Source code of SensitivityConvergenceCoverage class.
"""
import numpy as np

from mindspore.train.summary.summary_record import _get_summary_tensor_data
from mindarmour.fuzz_testing import CoverageMetrics
from mindarmour.utils._check_param import check_numpy_param
from mindarmour.utils.logger import LogUtil


LOGGER = LogUtil.get_instance()
TAG = 'CoverageMetrics'


class SensitivityConvergenceCoverage(CoverageMetrics):
    '''
    Get the metric of sensitivity convergence coverage: the proportion of neurons that have converged to the threshold.
    Sensitivity convergence coverage is a metric that can be used to evaluate the convergence of the neuron sensitivity
    to the threshold.

    Args:
        model (Model): Model to be evaluated.
        threshold (float): Threshold of sensitivity convergence coverage. Default: 0.001.
        incremental (bool): Whether to use incremental mode. Default: False.
        batch_size (int): Batch size. Default: 32.
        selected_neurons_num (int): Number of neurons selected for sensitivity convergence coverage. Default: 10.
    '''

    def __init__(self, model, threshold=0.5, incremental=False, batch_size=32, selected_neurons_num=100, n_iter=1000):
        super().__init__(model, incremental, batch_size)
        self.threshold = threshold
        self.total_converged = 0
        self.total_size = 0
        self.n_iter = n_iter
        self.selected_neurons_num = selected_neurons_num
        self.sensitive_neuron_idx = {}
        self.initial_samples = []

    def get_sensitive_neruon_idx(self, dataset):
        '''
        Args:
            dataset (numpy.ndarray): Dataset for evaluation.

        Returns:
            sensitive_neuron_idx(dict): The index of sensitive neurons.
        '''

        inputs = check_numpy_param('dataset', dataset)
        self._model.predict(Tensor(inputs))
        layer_out = _get_summary_tensor_data()
        for layer, tensor in layer_out.items():
            tensor = tensor.asnumpy().reshape(tensor.shape[0], -1)
            clean, benign = tensor[:tensor.shape[0] // 2], tensor[tensor.shape[0] // 2:]
            sensitivity = abs(clean-benign)
            self.sensitive_neuron_idx[layer] = np.argsort(np.sum(sensitivity,
                                                                 axis=0))[-min(self.selected_neurons_num, len(Helper)):]

    def get_metrics(self, dataset):
        '''
        Args:
            dataset (numpy.ndarray): Dataset for evaluation.

        Returns:
            SCC_value(float): the proportion of neurons that have converged to the threshold.

        Examples:
            >>> from mindspore.common.initializer import TruncatedNormal
            >>> from mindspore.ops import operations as P
            >>> from mindspore.train import Model
            >>> from mindspore.ops import TensorSummary
            >>> from mindarmour.fuzz_testing import NeuronCoverage
            >>> class Net(nn.Cell):
            ...     def __init__(self):
            ...         super(Net, self).__init__()
            ...         self.conv1 = nn.Conv2d(1, 6, 5, padding=0, weight_init=TruncatedNormal(0.02), pad_mode="valid")
            ...         self.conv2 = nn.Conv2d(6, 16, 5, padding=0, weight_init=TruncatedNormal(0.02), pad_mode="valid")
            ...         self.fc1 = nn.Dense(16 * 5 * 5, 120, TruncatedNormal(0.02), TruncatedNormal(0.02))
            ...         self.fc2 = nn.Dense(120, 84, TruncatedNormal(0.02), TruncatedNormal(0.02))
            ...         self.fc3 = nn.Dense(84, 10, TruncatedNormal(0.02), TruncatedNormal(0.02))
            ...         self.relu = nn.ReLU()
            ...         self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
            ...         self.reshape = P.Reshape()
            ...         self.summary = TensorSummary()
            ...     def construct(self, x):
            ...         x = self.conv1(x)
            ...         x = self.relu(x)
            ...         self.summary('conv1', x)
            ...         x = self.max_pool2d(x)
            ...         x = self.conv2(x)
            ...         x = self.relu(x)
            ...         self.summary('conv2', x)
            ...         x = self.max_pool2d(x)
            ...         x = self.reshape(x, (-1, 16 * 5 * 5))
            ...         x = self.fc1(x)
            ...         x = self.relu(x)
            ...         self.summary('fc1', x)
            ...         x = self.fc2(x)
            ...         x = self.relu(x)
            ...         self.summary('fc2', x)
            ...         x = self.fc3(x)
            ...         self.summary('fc3', x)
            ...         return x
            >>> net = Net()
            >>> model = Model(net)
            >>> batch_size = 32
            >>> num_classe = 10
            >>> test_images = np.random.rand(batch_size, 1, 32, 32).astype(np.float32)
            >>> test_images_adv = mutation(test_images)
            >>> test_set = np.concatenate((test_images, test_images_adv), axis=0)
            >>> SCC = SensitivityConvergenceCoverage(model,batch_size = batch_size*2)
            >>> SCC_metric = SCC.get_metrics(test_set)
        '''
        inputs = check_numpy_param('dataset', dataset)
        if not self.sensitive_neuron_idx:
            self.get_sensitive_neruon_idx(dataset)
        self._model.predict(Tensor(inputs))
        layer_out = _get_summary_tensor_data()
        for layer, tensor in layer_out.items():
            tensor = tensor.asnumpy().reshape(tensor.shape[0], -1)
            clean, benign = tensor[:tensor.shape[0] // 2], tensor[tensor.shape[0] // 2:]
            sensitivity = abs(clean-benign)
            try:
                sensitivity = sensitivity[:, self.sensitive_neuron_idx[layer]]
            except KeyError:
                raise RuntimeError('The layer {} is not in the sensitive_neuron_idx'.format(layer))
            converged, size = self.scc(sensitivity, sensitivity.shape[1], self.threshold)
            self.total_converged += converged
            self.total_size += size
        scc_value = self.total_converged/self.total_size
        return scc_value

    def scc(self, sensitivity_list, size, threshold=0):
        '''
        Args:
            sensitivity_list(numpy.ndarray): The sensitivity of each neuron.
            size(int): The number of neurons.
            threshold(float): The threshold of sensitivity convergence coverage.

        Returns:
            converged(int): The number of neurons that have converged to the threshold.
            size(int): The number of neurons.

        '''

        converged = 0
        for i in range(sensitivity_list.shape[1]):
            _, acceptance_rate = self.build_mh_chain(sensitivity_list[:, i],
                                                     np.mean(sensitivity_list[:, i]), self.n_iter, self.log_prob)

            if acceptance_rate > threshold:
                converged += 1

        return converged, size

    def proposal(self, x, stepsize):
        '''
        Args:
            x(numpy.ndarray): The input of the proposal function.
            stepsize(float): The stepsize of the proposal function.

        Returns:
            x(numpy.ndarray): The output of the proposal function.
        '''
        return np.random.uniform(low=x - 0.5 * stepsize,
                                 high=x + 0.5 * stepsize,
                                 size=x.shape)

    def p_acc_mh(self, x_new, x_old, log_prob):
        '''
        Args:
            x_new(numpy.ndarray): The new state.
            x_old(numpy.ndarray): The old state.
            log_prob(function): The log probability function.

        Returns:
            float: The acceptance probability.
        '''
        return min(1, np.exp(log_prob(x_new) - log_prob(x_old)))

    def log_prob(self, x):
        '''
        Args:
            x(numpy.ndarray): The input of the log probability function.

        Returns:
            float: The output of the log probability function.
        '''
        return -0.5 * np.sum(x ** 2)

    def sample_mh(self, x_old, log_prob, stepsize):
        '''
        here we determine whether we accept the new state or not:
        we draw a random number uniformly from [0,1] and compare
        it with the acceptance probability
        Args:
            x_old(numpy.ndarray): The old state.
            log_prob(function): The log probability function.
            stepsize(float): The stepsize of the proposal function.

        Returns:
            accept(bool): Whether to accept the new state.
            x_new(numpy.ndarray): The new state.
        '''
        x_new = self.proposal(x_old, stepsize)
        accept = np.random.random() < self.p_acc_mh(x_new, x_old, log_prob)
        if accept:
            return accept, x_new
        return accept, x_old

    def build_mh_chain(self, init, stepsize, n_total, log_prob):
        '''
        Args:
            init(numpy.ndarray): The initial state.
            stepsize(float): The stepsize of the proposal function.
            n_total(int): The total number of samples.
            log_prob(function): The log probability function.

        Returns:
            chain(list): The chain of samples.
            acceptance_rate(float): The acceptance rate of the chain.
        '''
        n_accepted = 0
        chain = [init]

        for _ in range(n_total):
            accept, state = self.sample_mh(chain[-1], log_prob, stepsize)
            chain.append(state)
            n_accepted += accept
        acceptance_rate = n_accepted / float(n_total)

        return chain, acceptance_rate

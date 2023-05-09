# Copyright 2023 Huawei Technologies Co., Ltd
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
Neuron sensitivity metrics.
"""

import mindspore as ms
from mindspore import ops

from mindarmour.utils.logger import LogUtil

# pylint: disable=redefined-builtin, unused-argument

LOGGER = LogUtil.get_instance()
TAG = "NeuronMetric"


class NeuronMetric:
    """
    Neuron sensitivity of models towards adversarial examples.

    Args:
        model (mindspore.nn.Cell): The victim model.
        inputs (mindspore.Tensor): Original samples.
        adv_inputs (mindspore.Tensor): Adversarial samples generated from original
            samples.
        hook_names (List[str]): The name of the evaluated layers.

    Raises:
        ValueError: If `output` is no more than 1 dimension.

    Examples:
        >>> from mindarmour.adv_robustness.evaluations import NeuronMetric
        >>> from mindspore import ops
        >>> # Refer to https://gitee.com/mindspore/docs/blob/master/docs/mindspore/code/lenet.py
        >>> model = LeNet()
        >>> x = ops.randn((10, 3, 32, 32))
        >>> adv_x = ops.randn((10, 3, 32, 32))
        >>> layers = ["conv1", "conv2", "fc1", "fc2"]
        >>> neuron_metric = NeuronMetric(model, x, adv_x, layers)
        >>> nsense = neuron_metric.neuron_sensitivity()
    """

    def __init__(self, model, inputs, adv_inputs, hook_names):
        self._model = model
        self._inputs = inputs
        self._adv_inputs = adv_inputs
        self._hook_names = hook_names

        hooks = []
        self._features = []

        def forward_hook(model, input, output):
            if len(output.shape) <= 1:
                msg = 'The output tensor should have more than 1 dimension, \
                    but got shape {}'.format(output.shape)
                LOGGER.error(msg)
                raise ValueError(msg)

            self._features.append(
                output.reshape(output.shape[0], output.shape[1], -1).mean(axis=-1)
            )

        for name in self._hook_names:
            hook = self._model.__getattr__(name).register_forward_hook(forward_hook)
            hooks.append(hook)

        self._model(ms.Tensor(self._inputs))
        self._model(ms.Tensor(self._adv_inputs))

        for hook in hooks:
            hook.remove()

    def neuron_sensitivity(self):
        """
        Calculate neuron sensitivity (NS).

        Returns:
            A dictionary, whose key is the layer name in hook_name,
            and the value is a numpy.ndarray of neuron sensitivity of each neuron.
        """
        nsense = {}
        n = len(self._hook_names)
        for i in range(n):
            nsense[self._hook_names[i]] = ops.norm(
                self._features[i] - self._features[i + n], axis=[0], p=1
            ).asnumpy()

        return nsense

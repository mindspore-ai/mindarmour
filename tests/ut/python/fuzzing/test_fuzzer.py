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
Model-fuzzer test.
"""
import numpy as np
import pytest
from mindspore import context
from mindspore import nn
from mindspore.common.initializer import TruncatedNormal
from mindspore.ops import operations as P
from mindspore.train import Model
from mindspore.ops import TensorSummary

from mindarmour.fuzz_testing import Fuzzer
from mindarmour.fuzz_testing import KMultisectionNeuronCoverage
from mindarmour.utils.logger import LogUtil

LOGGER = LogUtil.get_instance()
TAG = 'Fuzzing test'
LOGGER.set_level('INFO')


def conv(in_channels, out_channels, kernel_size, stride=1, padding=0):
    weight = weight_variable()
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=kernel_size, stride=stride, padding=padding,
                     weight_init=weight, has_bias=False, pad_mode="valid")


def fc_with_initialize(input_channels, out_channels):
    weight = weight_variable()
    bias = weight_variable()
    return nn.Dense(input_channels, out_channels, weight, bias)


def weight_variable():
    return TruncatedNormal(0.02)


class Net(nn.Cell):
    """
    Lenet network
    """

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = conv(1, 6, 5)
        self.conv2 = conv(6, 16, 5)
        self.fc1 = fc_with_initialize(16 * 5 * 5, 120)
        self.fc2 = fc_with_initialize(120, 84)
        self.fc3 = fc_with_initialize(84, 10)
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.reshape = P.Reshape()
        self.summary = TensorSummary()

    def construct(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        self.summary('conv1', x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        self.summary('conv2', x)
        x = self.max_pool2d(x)
        x = self.reshape(x, (-1, 16 * 5 * 5))
        x = self.fc1(x)
        x = self.relu(x)
        self.summary('fc1', x)
        x = self.fc2(x)
        x = self.relu(x)
        self.summary('fc2', x)
        x = self.fc3(x)
        self.summary('fc3', x)
        return x


@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
@pytest.mark.component_mindarmour
def test_fuzzing_ascend():
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    # load network
    net = Net()
    model = Model(net)
    batch_size = 8
    num_classe = 10
    mutate_config = [{'method': 'Blur',
                      'params': {'auto_param': [True]}},
                     {'method': 'Contrast',
                      'params': {'factor': [2, 1]}},
                     {'method': 'Translate',
                      'params': {'x_bias': [0.1, 0.3], 'y_bias': [0.2]}},
                     {'method': 'FGSM',
                      'params': {'eps': [0.1, 0.2, 0.3], 'alpha': [0.1]}}
                     ]

    train_images = np.random.rand(32, 1, 32, 32).astype(np.float32)
    # fuzz test with original test data
    # get test data
    test_images = np.random.rand(batch_size, 1, 32, 32).astype(np.float32)
    test_labels = np.random.randint(num_classe, size=batch_size).astype(np.int32)
    test_labels = (np.eye(num_classe)[test_labels]).astype(np.float32)

    initial_seeds = []
    # make initial seeds
    for img, label in zip(test_images, test_labels):
        initial_seeds.append([img, label])

    initial_seeds = initial_seeds[:100]

    nc = KMultisectionNeuronCoverage(model, train_images, segmented_num=100)
    cn_metrics = nc.get_metrics(test_images[:100])
    print('neuron coverage of initial seeds is: ', cn_metrics)
    model_fuzz_test = Fuzzer(model)
    _, _, _, _, metrics = model_fuzz_test.fuzzing(mutate_config, initial_seeds, nc, max_iters=100)
    print(metrics)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.component_mindarmour
def test_fuzzing_cpu():
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    # load network
    net = Net()
    model = Model(net)
    batch_size = 8
    num_classe = 10
    mutate_config = [{'method': 'Blur',
                      'params': {'auto_param': [True]}},
                     {'method': 'Contrast',
                      'params': {'factor': [2, 1]}},
                     {'method': 'Translate',
                      'params': {'x_bias': [0.1, 0.3], 'y_bias': [0.2]}},
                     {'method': 'FGSM',
                      'params': {'eps': [0.1, 0.2, 0.3], 'alpha': [0.1]}}
                     ]
    # initialize fuzz test with training dataset
    train_images = np.random.rand(32, 1, 32, 32).astype(np.float32)

    # fuzz test with original test data
    # get test data
    test_images = np.random.rand(batch_size, 1, 32, 32).astype(np.float32)
    test_labels = np.random.randint(num_classe, size=batch_size).astype(np.int32)
    test_labels = (np.eye(num_classe)[test_labels]).astype(np.float32)

    initial_seeds = []
    # make initial seeds
    for img, label in zip(test_images, test_labels):
        initial_seeds.append([img, label])

    initial_seeds = initial_seeds[:100]
    nc = KMultisectionNeuronCoverage(model, train_images, segmented_num=100)
    tknc_metrics = nc.get_metrics(test_images[:100])
    print('neuron coverage of initial seeds is: ', tknc_metrics)
    model_fuzz_test = Fuzzer(model)
    _, _, _, _, metrics = model_fuzz_test.fuzzing(mutate_config, initial_seeds, nc, max_iters=100)
    print(metrics)

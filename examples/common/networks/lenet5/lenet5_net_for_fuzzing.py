# Copyright 2021 Huawei Technologies Co., Ltd
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
lenet network with summary
"""
from mindspore import nn
from mindspore.common.initializer import TruncatedNormal
from mindspore.ops import TensorSummary


def conv(in_channels, out_channels, kernel_size, stride=1, padding=0):
    """Wrap conv."""
    weight = weight_variable()
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                     weight_init=weight, has_bias=False, pad_mode="valid")


def fc_with_initialize(input_channels, out_channels):
    """Wrap initialize method of full connection layer."""
    weight = weight_variable()
    bias = weight_variable()
    return nn.Dense(input_channels, out_channels, weight, bias)


def weight_variable():
    """Wrap initialize variable."""
    return TruncatedNormal(0.05)


class LeNet5(nn.Cell):
    """
    Lenet network
    """
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = conv(1, 6, 5)
        self.conv2 = conv(6, 16, 5)
        self.fc1 = fc_with_initialize(16*5*5, 120)
        self.fc2 = fc_with_initialize(120, 84)
        self.fc3 = fc_with_initialize(84, 10)
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.summary = TensorSummary()

    def construct(self, x):
        """
        construct the network architecture
        Returns:
            x (tensor): network output
        """
        x = self.conv1(x)
        self.summary('1', x)

        x = self.relu(x)
        self.summary('2', x)

        x = self.max_pool2d(x)
        self.summary('3', x)

        x = self.conv2(x)
        self.summary('4', x)

        x = self.relu(x)
        self.summary('5', x)

        x = self.max_pool2d(x)
        self.summary('6', x)

        x = self.flatten(x)
        self.summary('7', x)

        x = self.fc1(x)
        self.summary('8', x)

        x = self.relu(x)
        self.summary('9', x)

        x = self.fc2(x)
        self.summary('10', x)

        x = self.relu(x)
        self.summary('11', x)

        x = self.fc3(x)
        self.summary('output', x)
        return x

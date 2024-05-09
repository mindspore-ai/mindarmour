# Copyright 2024 Huawei Technologies Co., Ltd
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
CIFAR10CNN network.
"""
import collections
from mindspore import nn
from mindspore.common.initializer import TruncatedNormal

def weight_variable():
    return TruncatedNormal(0.05)

def conv(in_channels, out_channels, kernel_size, stride=1, padding=0):
    weight = weight_variable()
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=kernel_size, stride=stride, padding=padding,
                     weight_init=weight, has_bias=False, pad_mode="pad")


def fc_with_initialize(input_channels, out_channels):
    weight = weight_variable()
    bias = weight_variable()
    return nn.Dense(input_channels, out_channels, weight, bias)

class CIFAR10CNN(nn.Cell):
    """
    Part of CIFAR10CNN network.
    """
    def __init__(self):
        super(CIFAR10CNN, self).__init__()
        self.features = []
        self.layerdict = collections.OrderedDict()

        self.conv11 = conv(3, 64, 3, padding=1)
        self.relu11 = nn.ReLU()
        self.conv12 = conv(64, 64, 3, padding=1)
        self.relu12 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        self.features.extend([
            self.conv11, self.relu11, self.conv12, self.relu12, self.pool1
        ])
        self.layerdict['conv11'] = self.conv11
        self.layerdict['relu11'] = self.relu11
        self.layerdict['conv12'] = self.conv12
        self.layerdict['relu12'] = self.relu12
        self.layerdict['pool1'] = self.pool1

        self.conv21 = conv(64, 128, 3, padding=1)
        self.relu21 = nn.ReLU()
        self.conv22 = conv(128, 128, 3, padding=1)
        self.relu22 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.features.extend([
            self.conv21, self.relu21, self.conv22, self.relu22, self.pool2
        ])
        self.layerdict['conv21'] = self.conv21
        self.layerdict['relu21'] = self.relu21
        self.layerdict['conv22'] = self.conv22
        self.layerdict['relu22'] = self.relu22
        self.layerdict['pool2'] = self.pool2

        self.conv31 = conv(128, 128, 3, padding=1)
        self.relu31 = nn.ReLU()
        self.conv32 = conv(128, 128, 3, padding=1)
        self.relu32 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)
        self.features.extend([
            self.conv31, self.relu31, self.conv32, self.relu32, self.pool3
        ])
        self.layerdict['conv31'] = self.conv31
        self.layerdict['relu31'] = self.relu31
        self.layerdict['conv32'] = self.conv32
        self.layerdict['relu32'] = self.relu32
        self.layerdict['pool3'] = self.pool3

        self.classifier = []
        self.feature_dims = 4 * 4 * 128
        self.fc1 = fc_with_initialize(self.feature_dims, 512)
        self.fc1act = nn.Sigmoid()
        self.fc2 = fc_with_initialize(512, 10)
        self.classifier.extend([
            self.fc1, self.fc1act, self.fc2
        ])
        self.layerdict['fc1'] = self.fc1
        self.layerdict['fc1act'] = self.fc1act
        self.layerdict['fc2'] = self.fc2

    def construct(self, x):
        for layer in self.features:
            x = layer(x)
        x = x.view(-1, self.feature_dims)

        for layer in self.classifier:
            x = layer(x)
        return x

    def forward_from(self, x, layer):
        """
        Forward the network from the target layer.
        """
        if layer in self.layerdict:
            targetlayer = self.layerdict[layer]

            if targetlayer in self.features:
                layeridx = self.features.index(targetlayer)
                for func in self.features[layeridx + 1:]:
                    x = func(x)

                x = x.view(-1, self.feature_dims)
                for func in self.classifier:
                    x = func(x)
                return x
            layeridx = self.classifier.index(targetlayer)
            for func in self.classifier[layeridx:]:
                x = func(x)
            return x
        msg = "Target Layer not exists"
        raise ValueError(msg)

    def get_layer_output(self, x, targetlayer):
        """
        Get the output of the target layer.
        """
        targetlayer = self.layerdict.get(targetlayer, None)
        for layer in self.features:
            x = layer(x)
            if layer == targetlayer:
                return x
        x = x.view(-1, self.feature_dims)
        for layer in self.classifier:
            x = layer(x)
            if layer == targetlayer:
                return x
        msg = "Target Layer not exists"
        raise ValueError(msg)

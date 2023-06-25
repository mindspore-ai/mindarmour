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
A sample example for SCC.
"""
import numpy as np

import mindspore
from mindspore import Model
from mindspore import nn

from mindspore.dataset import vision, transforms
from mindspore.dataset import MnistDataset
from mindspore.common.initializer import TruncatedNormal
from mindspore.ops import operations as P
from mindspore.ops import TensorSummary
from mindarmour.fuzz_testing.fuzzing import SensitivityMaximizingFuzzer
from mindarmour.fuzz_testing.sensitivity_convergence_coverage import SensitivityConvergenceCoverage


def datapipe(path):
    """Prepare and return the MNIST dataset."""
    image_transforms = [
        vision.Rescale(1.0 / 255.0, 0),
        vision.HWC2CHW()
    ]
    label_transform = transforms.TypeCast(mindspore.int32)

    dataset = MnistDataset(path)
    dataset = dataset.map(image_transforms, 'image')
    dataset = dataset.map(label_transform, 'label')
    dataset = dataset.batch(100)
    return dataset


class Net(nn.Cell):
    """Convolutional neural network model."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 4, padding=0, weight_init=TruncatedNormal(0.02), pad_mode="valid")
        self.conv2 = nn.Conv2d(6, 16, 4, padding=0, weight_init=TruncatedNormal(0.02), pad_mode="valid")
        self.fc1 = nn.Dense(16 * 4 * 4, 120, TruncatedNormal(0.02), TruncatedNormal(0.02))
        self.fc2 = nn.Dense(120, 84, TruncatedNormal(0.02), TruncatedNormal(0.02))
        self.fc3 = nn.Dense(84, 10, TruncatedNormal(0.02), TruncatedNormal(0.02))
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.reshape = P.Reshape()
        self.summary = TensorSummary()

    def construct(self, x):
        """Construct the network."""
        x = self.conv1(x)
        x = self.relu(x)
        self.summary('conv1', x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        self.summary('conv2', x)
        x = self.max_pool2d(x)
        x = self.reshape(x, (-1, 16 * 4 * 4))
        x = self.fc1(x)
        x = self.relu(x)
        self.summary('fc1', x)
        x = self.fc2(x)
        x = self.relu(x)
        self.summary('fc2', x)
        x = self.fc3(x)
        self.summary('fc3', x)
        return x

model = Net()
param_dict = mindspore.load_checkpoint("model.ckpt")
mindspore.load_param_into_net(model, param_dict)

model = Model(model)
mutate_config = [{'method': 'GaussianBlur',
                  'params': {'ksize': [1, 2, 3, 5], 'auto_param': [True, False]}},
                 {'method': 'MotionBlur',
                  'params': {'degree': [1, 2, 5], 'angle': [45, 10, 100, 140, 210, 270, 300]}},
                 {'method': 'UniformNoise',
                  'params': {'factor': [0.1, 0.2, 0.3], 'auto_param': [False, True]}},
                 {'method': 'GaussianNoise',
                  'params': {'factor': [0.1, 0.2, 0.3], 'auto_param': [False, True]}},
                 {'method': 'Contrast',
                  'params': {'alpha': [0.5, 1, 1.5], 'beta': [-10, 0, 10], 'auto_param': [False, True]}},
                 {'method': 'Rotate',
                  'params': {'angle': [20, 90], 'auto_param': [False, True]}},
                 {'method': 'FGSM',
                  'params': {'eps': [0.3, 0.2, 0.4], 'alpha': [0.1], 'bounds': [(0, 1)]}}]

# make initial seeds
test_dataset = datapipe('MNIST_Data/test')


for data, label in test_dataset.create_tuple_iterator():
    initial_data = data
    initial_label = label
    break

initial_seeds = []
for data, label in test_dataset.create_tuple_iterator():
    initial_data = data
    initial_label = label
    break

for img, label in zip(initial_data, initial_label):
    label_array = np.array([0 if i != label else 1 for i in range(10)])
    initial_seeds.append([np.array(img).astype(np.float32), label_array.astype(np.float32)])

SCC = SensitivityConvergenceCoverage(model, batch_size=32)

print("SCC.get_metrics(initial_data)", SCC.get_metrics(initial_data))

model_fuzz_test = SensitivityMaximizingFuzzer(model)
samples, gt_labels, preds, strategies, metrics = model_fuzz_test.fuzzing(
    mutate_config, initial_seeds, SCC, max_iters=10)

print(metrics)

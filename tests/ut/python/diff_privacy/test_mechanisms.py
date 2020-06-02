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
different Privacy test.
"""
import pytest

from mindspore import context
from mindarmour.diff_privacy import GaussianRandom
from mindarmour.diff_privacy import AdaGaussianRandom
from mindarmour.diff_privacy import MechanismsFactory


@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.component_mindarmour
def test_gaussian():
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    shape = (3, 2, 4)
    norm_bound = 1.0
    initial_noise_multiplier = 0.1
    net = GaussianRandom(norm_bound, initial_noise_multiplier)
    res = net(shape)
    print(res)


@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.component_mindarmour
def test_ada_gaussian():
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    shape = (3, 2, 4)
    norm_bound = 1.0
    initial_noise_multiplier = 0.1
    noise_decay_rate = 0.5
    decay_policy = "Step"
    net = AdaGaussianRandom(norm_bound, initial_noise_multiplier,
                            noise_decay_rate, decay_policy)
    res = net(shape)
    print(res)


def test_factory():
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    shape = (3, 2, 4)
    norm_bound = 1.0
    initial_noise_multiplier = 0.1
    noise_decay_rate = 0.5
    decay_policy = "Step"
    noise_mechanism = MechanismsFactory()
    noise_construct = noise_mechanism.create('Gaussian',
                                             norm_bound,
                                             initial_noise_multiplier)
    noise = noise_construct(shape)
    print('Gaussian noise: ', noise)
    ada_mechanism = MechanismsFactory()
    ada_noise_construct = ada_mechanism.create('AdaGaussian',
                                               norm_bound,
                                               initial_noise_multiplier,
                                               noise_decay_rate,
                                               decay_policy)
    ada_noise = ada_noise_construct(shape)
    print('ada noise: ', ada_noise)


if __name__ == '__main__':
    # device_target can be "CPU", "GPU" or "Ascend"
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")

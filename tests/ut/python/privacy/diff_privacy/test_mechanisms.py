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
from mindspore import Tensor
from mindspore.common import dtype as mstype
from mindarmour.privacy.diff_privacy import NoiseAdaGaussianRandom
from mindarmour.privacy.diff_privacy import AdaClippingWithGaussianRandom
from mindarmour.privacy.diff_privacy import NoiseMechanismsFactory
from mindarmour.privacy.diff_privacy import ClipMechanismsFactory


@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
@pytest.mark.component_mindarmour
def test_graph_factory():
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    grad = Tensor([0.3, 0.2, 0.4], mstype.float32)
    norm_bound = 1.0
    initial_noise_multiplier = 0.1
    alpha = 0.5
    decay_policy = 'Step'
    factory = NoiseMechanismsFactory()
    noise_mech = factory.create('Gaussian',
                                norm_bound,
                                initial_noise_multiplier)
    noise = noise_mech(grad)
    print('Gaussian noise: ', noise)
    ada_noise_mech = factory.create('AdaGaussian',
                                    norm_bound,
                                    initial_noise_multiplier,
                                    noise_decay_rate=alpha,
                                    decay_policy=decay_policy)
    ada_noise = ada_noise_mech(grad)
    print('ada noise: ', ada_noise)


@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
@pytest.mark.component_mindarmour
def test_pynative_factory():
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    grad = Tensor([0.3, 0.2, 0.4], mstype.float32)
    norm_bound = 1.0
    initial_noise_multiplier = 0.1
    alpha = 0.5
    decay_policy = 'Step'
    factory = NoiseMechanismsFactory()
    noise_mech = factory.create('Gaussian',
                                norm_bound,
                                initial_noise_multiplier)
    noise = noise_mech(grad)
    print('Gaussian noise: ', noise)
    ada_noise_mech = factory.create('AdaGaussian',
                                    norm_bound,
                                    initial_noise_multiplier,
                                    noise_decay_rate=alpha,
                                    decay_policy=decay_policy)
    ada_noise = ada_noise_mech(grad)
    print('ada noise: ', ada_noise)


@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
@pytest.mark.component_mindarmour
def test_pynative_gaussian():
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    grad = Tensor([0.3, 0.2, 0.4], mstype.float32)
    norm_bound = 1.0
    initial_noise_multiplier = 0.1
    alpha = 0.5
    decay_policy = 'Step'
    factory = NoiseMechanismsFactory()
    noise_mech = factory.create('Gaussian',
                                norm_bound,
                                initial_noise_multiplier)
    noise = noise_mech(grad)
    print('Gaussian noise: ', noise)
    ada_noise_mech = factory.create('AdaGaussian',
                                    norm_bound,
                                    initial_noise_multiplier,
                                    noise_decay_rate=alpha,
                                    decay_policy=decay_policy)
    ada_noise = ada_noise_mech(grad)
    print('ada noise: ', ada_noise)


@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
@pytest.mark.component_mindarmour
def test_graph_ada_gaussian():
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    grad = Tensor([0.3, 0.2, 0.4], mstype.float32)
    norm_bound = 1.0
    initial_noise_multiplier = 0.1
    noise_decay_rate = 0.5
    decay_policy = 'Step'
    ada_noise_mech = NoiseAdaGaussianRandom(norm_bound,
                                            initial_noise_multiplier,
                                            seed=0,
                                            noise_decay_rate=noise_decay_rate,
                                            decay_policy=decay_policy)
    res = ada_noise_mech(grad)
    print(res)


@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
@pytest.mark.component_mindarmour
def test_pynative_ada_gaussian():
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    grad = Tensor([0.3, 0.2, 0.4], mstype.float32)
    norm_bound = 1.0
    initial_noise_multiplier = 0.1
    noise_decay_rate = 0.5
    decay_policy = 'Step'
    ada_noise_mech = NoiseAdaGaussianRandom(norm_bound,
                                            initial_noise_multiplier,
                                            seed=0,
                                            noise_decay_rate=noise_decay_rate,
                                            decay_policy=decay_policy)
    res = ada_noise_mech(grad)
    print(res)


@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
@pytest.mark.component_mindarmour
def test_graph_exponential():
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    grad = Tensor([0.3, 0.2, 0.4], mstype.float32)
    norm_bound = 1.0
    initial_noise_multiplier = 0.1
    alpha = 0.5
    decay_policy = 'Exp'
    factory = NoiseMechanismsFactory()
    ada_noise = factory.create('AdaGaussian',
                               norm_bound,
                               initial_noise_multiplier,
                               noise_decay_rate=alpha,
                               decay_policy=decay_policy)
    ada_noise = ada_noise(grad)
    print('ada noise: ', ada_noise)


@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
@pytest.mark.component_mindarmour
def test_pynative_exponential():
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    grad = Tensor([0.3, 0.2, 0.4], mstype.float32)
    norm_bound = 1.0
    initial_noise_multiplier = 0.1
    alpha = 0.5
    decay_policy = 'Exp'
    factory = NoiseMechanismsFactory()
    ada_noise = factory.create('AdaGaussian',
                               norm_bound,
                               initial_noise_multiplier,
                               noise_decay_rate=alpha,
                               decay_policy=decay_policy)
    ada_noise = ada_noise(grad)
    print('ada noise: ', ada_noise)


@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
@pytest.mark.component_mindarmour
def test_ada_clip_gaussian_random_pynative():
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    decay_policy = 'Linear'
    beta = Tensor(0.5, mstype.float32)
    norm_bound = Tensor(1.0, mstype.float32)
    beta_stddev = 0.1
    learning_rate = 0.1
    target_unclipped_quantile = 0.3
    ada_clip = AdaClippingWithGaussianRandom(decay_policy=decay_policy,
                                             learning_rate=learning_rate,
                                             target_unclipped_quantile=target_unclipped_quantile,
                                             fraction_stddev=beta_stddev,
                                             seed=1)
    next_norm_bound = ada_clip(beta, norm_bound)
    print('Liner next norm clip:', next_norm_bound)

    decay_policy = 'Geometric'
    ada_clip = AdaClippingWithGaussianRandom(decay_policy=decay_policy,
                                             learning_rate=learning_rate,
                                             target_unclipped_quantile=target_unclipped_quantile,
                                             fraction_stddev=beta_stddev,
                                             seed=1)
    next_norm_bound = ada_clip(beta, norm_bound)
    print('Geometric next norm clip:', next_norm_bound)


@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
@pytest.mark.component_mindarmour
def test_ada_clip_gaussian_random_graph():
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    decay_policy = 'Linear'
    beta = Tensor(0.5, mstype.float32)
    norm_bound = Tensor(1.0, mstype.float32)
    beta_stddev = 0.1
    learning_rate = 0.1
    target_unclipped_quantile = 0.3
    ada_clip = AdaClippingWithGaussianRandom(decay_policy=decay_policy,
                                             learning_rate=learning_rate,
                                             target_unclipped_quantile=target_unclipped_quantile,
                                             fraction_stddev=beta_stddev,
                                             seed=1)
    next_norm_bound = ada_clip(beta, norm_bound)
    print('Liner next norm clip:', next_norm_bound)

    decay_policy = 'Geometric'
    ada_clip = AdaClippingWithGaussianRandom(decay_policy=decay_policy,
                                             learning_rate=learning_rate,
                                             target_unclipped_quantile=target_unclipped_quantile,
                                             fraction_stddev=beta_stddev,
                                             seed=1)
    next_norm_bound = ada_clip(beta, norm_bound)
    print('Geometric next norm clip:', next_norm_bound)


@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
@pytest.mark.component_mindarmour
def test_pynative_clip_mech_factory():
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    decay_policy = 'Linear'
    beta = Tensor(0.5, mstype.float32)
    norm_bound = Tensor(1.0, mstype.float32)
    beta_stddev = 0.1
    learning_rate = 0.1
    target_unclipped_quantile = 0.3
    factory = ClipMechanismsFactory()
    ada_clip = factory.create('Gaussian',
                              decay_policy=decay_policy,
                              learning_rate=learning_rate,
                              target_unclipped_quantile=target_unclipped_quantile,
                              fraction_stddev=beta_stddev)
    next_norm_bound = ada_clip(beta, norm_bound)
    print('next_norm_bound: ', next_norm_bound)


@pytest.mark.level0
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
@pytest.mark.component_mindarmour
def test_graph_clip_mech_factory():
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    decay_policy = 'Linear'
    beta = Tensor(0.5, mstype.float32)
    norm_bound = Tensor(1.0, mstype.float32)
    beta_stddev = 0.1
    learning_rate = 0.1
    target_unclipped_quantile = 0.3
    factory = ClipMechanismsFactory()
    ada_clip = factory.create('Gaussian',
                              decay_policy=decay_policy,
                              learning_rate=learning_rate,
                              target_unclipped_quantile=target_unclipped_quantile,
                              fraction_stddev=beta_stddev)
    next_norm_bound = ada_clip(beta, norm_bound)
    print('next_norm_bound: ', next_norm_bound)

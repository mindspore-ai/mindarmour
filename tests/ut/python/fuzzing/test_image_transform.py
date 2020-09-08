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
Image transform test.
"""
import numpy as np
import pytest

from mindarmour.utils.logger import LogUtil
from mindarmour.fuzz_testing.image_transform import Contrast, Brightness, \
    Blur, Noise, Translate, Scale, Shear, Rotate

LOGGER = LogUtil.get_instance()
TAG = 'Image transform test'
LOGGER.set_level('INFO')


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.component_mindarmour
def test_contrast():
    image = (np.random.rand(32, 32)).astype(np.float32)
    trans = Contrast()
    trans.set_params(auto_param=True)
    _ = trans.transform(image)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.component_mindarmour
def test_brightness():
    image = (np.random.rand(32, 32)).astype(np.float32)
    trans = Brightness()
    trans.set_params(auto_param=True)
    _ = trans.transform(image)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
@pytest.mark.component_mindarmour
def test_blur():
    image = (np.random.rand(32, 32)).astype(np.float32)
    trans = Blur()
    trans.set_params(auto_param=True)
    _ = trans.transform(image)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
@pytest.mark.component_mindarmour
def test_noise():
    image = (np.random.rand(32, 32)).astype(np.float32)
    trans = Noise()
    trans.set_params(auto_param=True)
    _ = trans.transform(image)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
@pytest.mark.component_mindarmour
def test_translate():
    image = (np.random.rand(32, 32)).astype(np.float32)
    trans = Translate()
    trans.set_params(auto_param=True)
    _ = trans.transform(image)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
@pytest.mark.component_mindarmour
def test_shear():
    image = (np.random.rand(32, 32)).astype(np.float32)
    trans = Shear()
    trans.set_params(auto_param=True)
    _ = trans.transform(image)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
@pytest.mark.component_mindarmour
def test_scale():
    image = (np.random.rand(32, 32)).astype(np.float32)
    trans = Scale()
    trans.set_params(auto_param=True)
    _ = trans.transform(image)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
@pytest.mark.component_mindarmour
def test_rotate():
    image = (np.random.rand(32, 32)).astype(np.float32)
    trans = Rotate()
    trans.set_params(auto_param=True)
    _ = trans.transform(image)

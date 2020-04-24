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
from mindarmour.utils.image_transform import Contrast, Brightness, Blur, Noise, \
    Translate, Scale, Shear, Rotate

LOGGER = LogUtil.get_instance()
TAG = 'Image transform test'
LOGGER.set_level('INFO')


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.component_mindarmour
def test_contrast():
    image = (np.random.rand(32, 32)*255).astype(np.float32)
    mode = 'L'
    trans = Contrast(image, mode)
    trans.random_param()
    trans_image = trans.transform()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.component_mindarmour
def test_brightness():
    image = (np.random.rand(32, 32)*255).astype(np.float32)
    mode = 'L'
    trans = Brightness(image, mode)
    trans.random_param()
    trans_image = trans.transform()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
@pytest.mark.component_mindarmour
def test_blur():
    image = (np.random.rand(32, 32)*255).astype(np.float32)
    mode = 'L'
    trans = Blur(image, mode)
    trans.random_param()
    trans_image = trans.transform()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
@pytest.mark.component_mindarmour
def test_noise():
    image = (np.random.rand(32, 32)*255).astype(np.float32)
    mode = 'L'
    trans = Noise(image, mode)
    trans.random_param()
    trans_image = trans.transform()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
@pytest.mark.component_mindarmour
def test_translate():
    image = (np.random.rand(32, 32)*255).astype(np.float32)
    mode = 'L'
    trans = Translate(image, mode)
    trans.random_param()
    trans_image = trans.transform()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
@pytest.mark.component_mindarmour
def test_shear():
    image = (np.random.rand(32, 32)*255).astype(np.float32)
    mode = 'L'
    trans = Shear(image, mode)
    trans.random_param()
    trans_image = trans.transform()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
@pytest.mark.component_mindarmour
def test_scale():
    image = (np.random.rand(32, 32)*255).astype(np.float32)
    mode = 'L'
    trans = Scale(image, mode)
    trans.random_param()
    trans_image = trans.transform()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
@pytest.mark.component_mindarmour
def test_rotate():
    image = (np.random.rand(32, 32)*255).astype(np.float32)
    mode = 'L'
    trans = Rotate(image, mode)
    trans.random_param()
    trans_image = trans.transform()



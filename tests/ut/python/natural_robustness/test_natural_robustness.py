# Copyright 2022 Huawei Technologies Co., Ltd
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

"""Example for natural robustness methods."""

import pytest
import numpy as np
from mindspore import context

from mindarmour.natural_robustness.image import Translate, Curve, Perspective, Scale, Shear, Rotate, SaltAndPepperNoise, \
    NaturalNoise, GaussianNoise, UniformNoise, MotionBlur, GaussianBlur, GradientBlur, Contrast, GradientLuminance


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_perspective():
    """
    Feature: Test image perspective.
    Description: Image will be transform for given perspective projection.
    Expectation: success.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    image = np.random.random((32, 32, 3))
    ori_pos = [[0, 0], [0, 800], [800, 0], [800, 800]]
    dst_pos = [[50, 0], [0, 800], [780, 0], [800, 800]]
    trans = Perspective(ori_pos, dst_pos)
    dst = trans(image)
    print(dst)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_uniform_noise():
    """
    Feature: Test image uniform noise.
    Description: Add uniform image in image.
    Expectation: success.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    image = np.random.random((32, 32, 3))
    trans = UniformNoise(factor=0.1)
    dst = trans(image)
    print(dst)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_gaussian_noise():
    """
    Feature: Test image gaussian noise.
    Description: Add gaussian image in image.
    Expectation: success.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    image = np.random.random((32, 32, 3))
    trans = GaussianNoise(factor=0.1)
    dst = trans(image)
    print(dst)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_contrast():
    """
    Feature: Test image contrast.
    Description: Adjust image contrast.
    Expectation: success.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    image = np.random.random((32, 32, 3))
    trans = Contrast(alpha=0.3, beta=0)
    dst = trans(image)
    print(dst)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_gaussian_blur():
    """
    Feature: Test image gaussian blur.
    Description: Add gaussian blur to image.
    Expectation: success.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    image = np.random.random((32, 32, 3))
    trans = GaussianBlur(ksize=5)
    dst = trans(image)
    print(dst)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_salt_and_pepper_noise():
    """
    Feature: Test image salt and pepper noise.
    Description: Add salt and pepper to image.
    Expectation: success.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    image = np.random.random((32, 32, 3))
    trans = SaltAndPepperNoise(factor=0.01)
    dst = trans(image)
    print(dst)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_translate():
    """
    Feature: Test image translate.
    Description: Translate an image.
    Expectation: success.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    image = np.random.random((32, 32, 3))
    trans = Translate(x_bias=0.1, y_bias=0.1)
    dst = trans(image)
    print(dst)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_scale():
    """
    Feature: Test image scale.
    Description: Scale an image.
    Expectation: success.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    image = np.random.random((32, 32, 3))
    trans = Scale(factor_x=0.7, factor_y=0.7)
    dst = trans(image)
    print(dst)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_shear():
    """
    Feature: Test image shear.
    Description: Shear an image.
    Expectation: success.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    image = np.random.random((32, 32, 3))
    trans = Shear(factor=0.2)
    dst = trans(image)
    print(dst)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_rotate():
    """
    Feature: Test image rotate.
    Description: Rotate an image.
    Expectation: success.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    image = np.random.random((32, 32, 3))
    trans = Rotate(angle=20)
    dst = trans(image)
    print(dst)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_curve():
    """
    Feature: Test image curve.
    Description: Transform an image with curve.
    Expectation: success.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    image = np.random.random((32, 32, 3))
    trans = Curve(curves=1.5, depth=1.5, mode='horizontal')
    dst = trans(image)
    print(dst)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_natural_noise():
    """
    Feature: Test natural noise.
    Description: Add natural noise to an.
    Expectation: success.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    image = np.random.random((32, 32, 3))
    trans = NaturalNoise(ratio=0.0001, k_x_range=(1, 30), k_y_range=(1, 10), auto_param=True)
    dst = trans(image)
    print(dst)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_gradient_luminance():
    """
    Feature: Test gradient luminance.
    Description: Adjust image luminance.
    Expectation: success.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    image = np.random.random((32, 32, 3))
    height, width = image.shape[:2]
    point = (height // 4, width // 2)
    start = (255, 255, 255)
    end = (0, 0, 0)
    scope = 0.3
    bright_rate = 0.4
    trans = GradientLuminance(start, end, start_point=point, scope=scope, pattern='dark', bright_rate=bright_rate,
                              mode='horizontal')
    dst = trans(image)
    print(dst)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_motion_blur():
    """
    Feature: Test motion blur.
    Description: Add motion blur to an image.
    Expectation: success.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    image = np.random.random((32, 32, 3))
    angle = -10.5
    i = 3
    trans = MotionBlur(degree=i, angle=angle)
    dst = trans(image)
    print(dst)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_gradient_blur():
    """
    Feature: Test gradient blur.
    Description: Add gradient blur to an image.
    Expectation: success.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    image = np.random.random((32, 32, 3))
    number = 10
    h, w = image.shape[:2]
    point = (int(h / 5), int(w / 5))
    center = False
    trans = GradientBlur(point, number, center)
    dst = trans(image)
    print(dst)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_perspective_ascend():
    """
    Feature: Test image perspective.
    Description: Image will be transform for given perspective projection.
    Expectation: success.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    image = np.random.random((32, 32, 3))
    ori_pos = [[0, 0], [0, 800], [800, 0], [800, 800]]
    dst_pos = [[50, 0], [0, 800], [780, 0], [800, 800]]
    trans = Perspective(ori_pos, dst_pos)
    dst = trans(image)
    print(dst)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_uniform_noise_ascend():
    """
    Feature: Test image uniform noise.
    Description: Add uniform image in image.
    Expectation: success.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    image = np.random.random((32, 32, 3))
    trans = UniformNoise(factor=0.1)
    dst = trans(image)
    print(dst)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_gaussian_noise_ascend():
    """
    Feature: Test image gaussian noise.
    Description: Add gaussian image in image.
    Expectation: success.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    image = np.random.random((32, 32, 3))
    trans = GaussianNoise(factor=0.1)
    dst = trans(image)
    print(dst)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_contrast_ascend():
    """
    Feature: Test image contrast.
    Description: Adjust image contrast.
    Expectation: success.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    image = np.random.random((32, 32, 3))
    trans = Contrast(alpha=0.3, beta=0)
    dst = trans(image)
    print(dst)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_gaussian_blur_ascend():
    """
    Feature: Test image gaussian blur.
    Description: Add gaussian blur to image.
    Expectation: success.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    image = np.random.random((32, 32, 3))
    trans = GaussianBlur(ksize=5)
    dst = trans(image)
    print(dst)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_salt_and_pepper_noise_ascend():
    """
    Feature: Test image salt and pepper noise.
    Description: Add salt and pepper to image.
    Expectation: success.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    image = np.random.random((32, 32, 3))
    trans = SaltAndPepperNoise(factor=0.01)
    dst = trans(image)
    print(dst)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_translate_ascend():
    """
    Feature: Test image translate.
    Description: Translate an image.
    Expectation: success.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    image = np.random.random((32, 32, 3))
    trans = Translate(x_bias=0.1, y_bias=0.1)
    dst = trans(image)
    print(dst)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_card
@pytest.mark.component_ascend_mindarmour
def test_scale_ascend():
    """
    Feature: Test image scale.
    Description: Scale an image.
    Expectation: success.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    image = np.random.random((32, 32, 3))
    trans = Scale(factor_x=0.7, factor_y=0.7)
    dst = trans(image)
    print(dst)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_shear_ascend():
    """
    Feature: Test image shear.
    Description: Shear an image.
    Expectation: success.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    image = np.random.random((32, 32, 3))
    trans = Shear(factor=0.2)
    dst = trans(image)
    print(dst)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_rotate_ascend():
    """
    Feature: Test image rotate.
    Description: Rotate an image.
    Expectation: success.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    image = np.random.random((32, 32, 3))
    trans = Rotate(angle=20)
    dst = trans(image)
    print(dst)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_curve_ascend():
    """
    Feature: Test image curve.
    Description: Transform an image with curve.
    Expectation: success.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    image = np.random.random((32, 32, 3))
    trans = Curve(curves=1.5, depth=1.5, mode='horizontal')
    dst = trans(image)
    print(dst)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_natural_noise_ascend():
    """
    Feature: Test natural noise.
    Description: Add natural noise to an.
    Expectation: success.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    image = np.random.random((32, 32, 3))
    trans = NaturalNoise(ratio=0.0001, k_x_range=(1, 30), k_y_range=(1, 10), auto_param=True)
    dst = trans(image)
    print(dst)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_gradient_luminance_ascend():
    """
    Feature: Test gradient luminance.
    Description: Adjust image luminance.
    Expectation: success.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    image = np.random.random((32, 32, 3))
    height, width = image.shape[:2]
    point = (height // 4, width // 2)
    start = (255, 255, 255)
    end = (0, 0, 0)
    scope = 0.3
    bright_rate = 0.4
    trans = GradientLuminance(start, end, start_point=point, scope=scope, pattern='dark', bright_rate=bright_rate,
                              mode='horizontal')
    dst = trans(image)
    print(dst)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_motion_blur_ascend():
    """
    Feature: Test motion blur.
    Description: Add motion blur to an image.
    Expectation: success.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    image = np.random.random((32, 32, 3))
    angle = -10.5
    i = 3
    trans = MotionBlur(degree=i, angle=angle)
    dst = trans(image)
    print(dst)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_card
@pytest.mark.component_mindarmour
def test_gradient_blur_ascend():
    """
    Feature: Test gradient blur.
    Description: Add gradient blur to an image.
    Expectation: success.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    image = np.random.random((32, 32, 3))
    number = 10
    h, w = image.shape[:2]
    point = (int(h / 5), int(w / 5))
    center = False
    trans = GradientBlur(point, number, center)
    dst = trans(image)
    print(dst)

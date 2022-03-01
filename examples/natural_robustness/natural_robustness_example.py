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

import numpy as np
import cv2

from mindarmour.natural_robustness.transform.image import Translate, Curve, Perspective, Scale, Shear, Rotate, SaltAndPepperNoise, \
    NaturalNoise, GaussianNoise, UniformNoise, MotionBlur, GaussianBlur, GradientBlur, Contrast, GradientLuminance


def test_perspective(image):
    """Test perspective."""
    ori_pos = [[0, 0], [0, 800], [800, 0], [800, 800]]
    dst_pos = [[50, 0], [0, 800], [780, 0], [800, 800]]
    trans = Perspective(ori_pos, dst_pos)
    dst = trans(image)
    cv2.imshow('dst', dst)
    cv2.waitKey()


def test_uniform_noise(image):
    """Test uniform noise."""
    trans = UniformNoise(factor=0.1)
    dst = trans(image)
    cv2.imshow('dst', dst)
    cv2.waitKey()


def test_gaussian_noise(image):
    """Test gaussian noise."""
    trans = GaussianNoise(factor=0.1)
    dst = trans(image)
    cv2.imshow('dst', dst)
    cv2.waitKey()


def test_contrast(image):
    """Test contrast."""
    trans = Contrast(alpha=2, beta=0)
    dst = trans(image)
    cv2.imshow('dst', dst)
    cv2.waitKey()


def test_gaussian_blur(image):
    """Test gaussian blur."""
    trans = GaussianBlur(ksize=5)
    dst = trans(image)
    cv2.imshow('dst', dst)
    cv2.waitKey()


def test_salt_and_pepper_noise(image):
    """Test salt and pepper noise."""
    trans = SaltAndPepperNoise(factor=0.01)
    dst = trans(image)
    cv2.imshow('dst', dst)
    cv2.waitKey()


def test_translate(image):
    """Test translate."""
    trans = Translate(x_bias=0.1, y_bias=0.1)
    dst = trans(image)
    cv2.imshow('dst', dst)
    cv2.waitKey()


def test_scale(image):
    """Test scale."""
    trans = Scale(factor_x=0.7, factor_y=0.7)
    dst = trans(image)
    cv2.imshow('dst', dst)
    cv2.waitKey()


def test_shear(image):
    """Test shear."""
    trans = Shear(factor=0.2)
    dst = trans(image)
    cv2.imshow('dst', dst)
    cv2.waitKey()


def test_rotate(image):
    """Test rotate."""
    trans = Rotate(angle=20)
    dst = trans(image)
    cv2.imshow('dst', dst)
    cv2.waitKey()


def test_curve(image):
    """Test curve."""
    trans = Curve(curves=2, depth=1.5, mode='horizontal')
    dst = trans(image)
    cv2.imshow('dst', dst)
    cv2.waitKey()


def test_natural_noise(image):
    """Test natural noise."""
    trans = NaturalNoise(ratio=0.0001, k_x_range=(1, 30), k_y_range=(1, 10), auto_param=True)
    dst = trans(image)
    cv2.imshow('dst', dst)
    cv2.waitKey()


def test_gradient_luminance(image):
    """Test gradient luminance."""
    height, width = image.shape[:2]
    point = (height // 4, width // 2)
    start = (255, 255, 255)
    end = (0, 0, 0)
    scope = 0.3
    bright_rate = 0.4
    trans = GradientLuminance(start, end, start_point=point, scope=scope, pattern='dark', bright_rate=bright_rate,
                              mode='horizontal')
    dst = trans(image)
    cv2.imshow('dst', dst)
    cv2.waitKey()


def test_motion_blur(image):
    """Test motion blur."""
    angle = -10.5
    i = 10
    trans = MotionBlur(degree=i, angle=angle)
    dst = trans(image)
    cv2.imshow('dst', dst)
    cv2.waitKey()


def test_gradient_blur(image):
    """Test gradient blur."""
    number = 10
    h, w = image.shape[:2]
    point = (int(h / 5), int(w / 5))
    center = False
    trans = GradientBlur(point, number, center)
    dst = trans(image)
    cv2.imshow('dst', dst)
    cv2.waitKey()


if __name__ == '__main__':
    img = cv2.imread('1.jpeg')
    img = np.array(img)
    test_uniform_noise(img)
    test_gaussian_noise(img)
    test_motion_blur(img)
    test_gradient_blur(img)
    test_gradient_luminance(img)
    test_natural_noise(img)
    test_curve(img)
    test_rotate(img)
    test_shear(img)
    test_scale(img)
    test_translate(img)
    test_salt_and_pepper_noise(img)
    test_gaussian_blur(img)
    test_contrast(img)
    test_perspective(img)

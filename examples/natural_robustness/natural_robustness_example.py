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

"""Example for natural robustness methods."""

import numpy as np
import cv2

from mindarmour.natural_robustness.natural_noise import Perlin, Perspective, Scale, Shear, SaltAndPepperNoise, \
    BackgroundWord, BackShadow, MotionBlur, GaussianBlur, GradientBlur, Rotate, Contrast, Translate, Curve, \
    GradientLuminance, NaturalNoise


def test_perspective(image):
    ori_pos = [[0, 0], [0, 800], [800, 0], [800, 800]]
    dst_pos = [[50, 0], [0, 800], [780, 0], [800, 800]]
    trans = Perspective(ori_pos, dst_pos)
    dst = trans(image)
    cv2.imshow('dst', dst)
    cv2.waitKey()


def test_constract(image):
    trans = Contrast(alpha=1.5, beta=0)
    dst = trans(image)
    cv2.imshow('dst', dst)
    cv2.waitKey()


def test_gaussian_blur(image):
    trans = GaussianBlur(ksize=5)
    dst = trans(image)
    cv2.imshow('dst', dst)
    cv2.waitKey()


def test_salt_and_pepper_noise(image):
    trans = SaltAndPepperNoise(factor=0.01)
    dst = trans(image)
    cv2.imshow('dst', dst)
    cv2.waitKey()


def test_translate(image):
    trans = Translate(x_bias=0.1, y_bias=0.1)
    dst = trans(image)
    cv2.imshow('dst', dst)
    cv2.waitKey()


def test_scale(image):
    trans = Scale(factor_x=0.7, factor_y=0.7)
    dst = trans(image)
    cv2.imshow('dst', dst)
    cv2.waitKey()


def test_shear(image):
    trans = Shear(factor=0.2)
    dst = trans(image)
    cv2.imshow('dst', dst)
    cv2.waitKey()


def test_rotate(image):
    trans = Rotate(angle=20)
    dst = trans(image)
    cv2.imshow('dst', dst)
    cv2.waitKey()


def test_background_word(image):
    trans = BackgroundWord(shade=0.1)
    dst = trans(image)
    cv2.imshow('dst', dst)
    cv2.waitKey()


def test_curve(image):
    trans = Curve(curves=1.5, depth=1.5, mode='horizontal')
    dst = trans(image)
    cv2.imshow('dst', dst)
    cv2.waitKey()


def test_natural_noise(image):
    trans = NaturalNoise(ratio=0.0001, k_x_range=(1, 30), k_y_range=(1, 10))
    dst = trans(image)
    cv2.imshow('dst', dst)
    cv2.waitKey()


def test_back_shadow(image):
    image = np.array(image)
    template_path = 'test_data/template/leaf'
    shade = 0.2
    trans = BackShadow(template_path, shade=shade)
    dst = trans(image)
    cv2.imshow('dst', dst)
    cv2.waitKey()


def test_perlin(image):
    image = np.array(image)
    shade = 0.5
    ratio = 0.3
    trans = Perlin(ratio, shade)
    dst = trans(image)
    cv2.imshow('dst', dst)
    cv2.waitKey()


def test_gradient_luminance(image):
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
    angle = -10.5
    i = 3
    trans = MotionBlur(degree=i, angle=angle)
    dst = trans(image)
    cv2.imshow('dst', dst)
    cv2.waitKey()


def test_gradient_blur(image):
    number = 4
    h, w = image.shape[:2]
    point = (int(h / 5), int(w / 5))
    center = True
    trans = GradientBlur(point, number, center)
    dst = trans(image)
    cv2.imshow('dst', dst)
    cv2.waitKey()


if __name__ == '__main__':
    img = cv2.imread('test_data/1.png')
    img = np.array(img)
    test_motion_blur(img)
    test_gradient_blur(img)
    test_gradient_luminance(img)
    test_perlin(img)
    test_back_shadow(img)
    test_natural_noise(img)
    test_curve(img)
    test_background_word(img)
    test_rotate(img)
    test_shear(img)
    test_scale(img)
    test_translate(img)
    test_salt_and_pepper_noise(img)
    test_gaussian_blur(img)
    test_constract(img)
    test_perspective(img)

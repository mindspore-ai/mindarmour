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
"""
Image corruption.
"""
import math
import numpy as np
import cv2

from mindarmour.natural_robustness.transform.image.natural_perturb import _NaturalPerturb
from mindarmour.utils._check_param import check_param_multi_types, check_param_type
from mindarmour.utils.logger import LogUtil

LOGGER = LogUtil.get_instance()
TAG = 'Image corruption'


class UniformNoise(_NaturalPerturb):
    """
    Add uniform noise of an image.

    Args:
        factor (float): Noise density, the proportion of noise points per unit pixel area. Suggested value range in
            [0.001, 0.15].
        auto_param (bool): Auto selected parameters. Selected parameters will preserve semantics of image.

    Example:
        >>> img = cv2.imread('1.png')
        >>> img = np.array(img)
        >>> factor = 0.1
        >>> trans = UniformNoise(factor)
        >>> dst = trans(img)
    """

    def __init__(self, factor=0.1, auto_param=False):
        super(UniformNoise, self).__init__()
        self.factor = check_param_multi_types('factor', factor, [int, float])
        check_param_type('auto_param', auto_param, bool)
        if auto_param:
            self.factor = np.random.uniform(0, 0.15)

    def __call__(self, image):
        """
        Transform the image.

        Args:
            image (numpy.ndarray): Original image to be transformed.

        Returns:
            numpy.ndarray, transformed image.
        """
        ori_dtype = image.dtype
        _, chw, normalized, gray3dim, image = self._check(image)
        low, high = (0, 255)
        weight = self.factor * (high - low)
        noise = np.random.uniform(-weight, weight, size=image.shape)
        trans_image = np.clip(image + noise, low, high)
        trans_image = self._original_format(trans_image, chw, normalized, gray3dim)

        return trans_image.astype(ori_dtype)


class GaussianNoise(_NaturalPerturb):
    """
    Add gaussian noise of an image.

    Args:
        factor (float): Noise density, the proportion of noise points per unit pixel area. Suggested value range in
            [0.001, 0.15].
        auto_param (bool): Auto selected parameters. Selected parameters will preserve semantics of image.

    Example:
        >>> img = cv2.imread('1.png')
        >>> img = np.array(img)
        >>> factor = 0.1
        >>> trans = GaussianNoise(factor)
        >>> dst = trans(img)
    """

    def __init__(self, factor=0.1, auto_param=False):
        super(GaussianNoise, self).__init__()
        self.factor = check_param_multi_types('factor', factor, [int, float])
        check_param_type('auto_param', auto_param, bool)
        if auto_param:
            self.factor = np.random.uniform(0, 0.15)

    def __call__(self, image):
        """
        Transform the image.

        Args:
            image (numpy.ndarray): Original image to be transformed.

        Returns:
            numpy.ndarray, transformed image.
        """
        ori_dtype = image.dtype
        low, high = (0, 255)
        _, chw, normalized, gray3dim, image = self._check(image)
        std = self.factor / math.sqrt(3) * (high - low)
        noise = np.random.normal(scale=std, size=image.shape)
        trans_image = np.clip(image + noise, low, high)
        trans_image = self._original_format(trans_image, chw, normalized, gray3dim)
        return trans_image.astype(ori_dtype)


class SaltAndPepperNoise(_NaturalPerturb):
    """
    Add salt and pepper noise of an image.

    Args:
        factor (float): Noise density, the proportion of noise points per unit pixel area. Suggested value range in
            [0.001, 0.15].
        auto_param (bool): Auto selected parameters. Selected parameters will preserve semantics of image.

    Example:
        >>> img = cv2.imread('1.png')
        >>> img = np.array(img)
        >>> factor = 0.1
        >>> trans = SaltAndPepperNoise(factor)
        >>> dst = trans(img)
    """

    def __init__(self, factor=0, auto_param=False):
        super(SaltAndPepperNoise, self).__init__()
        self.factor = check_param_multi_types('factor', factor, [int, float])
        check_param_type('auto_param', auto_param, bool)
        if auto_param:
            self.factor = np.random.uniform(0, 0.15)

    def __call__(self, image):
        """
        Transform the image.

        Args:
            image (numpy.ndarray): Original image to be transformed.

        Returns:
            numpy.ndarray, transformed image.
        """
        ori_dtype = image.dtype
        _, chw, normalized, gray3dim, image = self._check(image)
        low, high = (0, 255)
        noise = np.random.uniform(low=-1, high=1, size=(image.shape[0], image.shape[1]))
        trans_image = np.copy(image)
        threshold = 1 - self.factor
        trans_image[noise < -threshold] = low
        trans_image[noise > threshold] = high
        trans_image = self._original_format(trans_image, chw, normalized, gray3dim)
        return trans_image.astype(ori_dtype)


class NaturalNoise(_NaturalPerturb):
    """
    Add natural noise to an image.

    Args:
        ratio (float): Noise density, the proportion of noise blocks per unit pixel area. Suggested value range in
            [0.00001, 0.001].
        k_x_range (union[list, tuple]): Value range of the noise block length.
        k_y_range (union[list, tuple]): Value range of the noise block width.
        auto_param (bool): Auto selected parameters. Selected parameters will preserve semantics of image.

    Examples:
        >>> img = cv2.imread('xx.png')
        >>> img = np.array(img)
        >>> ratio = 0.0002
        >>> k_x_range = (1, 5)
        >>> k_y_range = (3, 25)
        >>> trans = NaturalNoise(ratio, k_x_range, k_y_range)
        >>> new_img = trans(img)
    """

    def __init__(self, ratio=0.0002, k_x_range=(1, 5), k_y_range=(3, 25), auto_param=False):
        super(NaturalNoise).__init__()
        self.ratio = check_param_type('ratio', ratio, float)
        k_x_range = check_param_multi_types('k_x_range', k_x_range, [list, tuple])
        k_y_range = check_param_multi_types('k_y_range', k_y_range, [list, tuple])
        self.k_x_range = tuple(k_x_range)
        self.k_y_range = tuple(k_y_range)
        self.auto_param = check_param_type('auto_param', auto_param, bool)

    def __call__(self, image):
        """
        Add natural noise to given image.

        Args:
            image (numpy.ndarray): Original image.

        Returns:
            numpy.ndarray, image with natural noise.
        """
        ori_dtype = image.dtype
        _, chw, normalized, gray3dim, image = self._check(image)
        randon_range = 100
        w, h = image.shape[:2]
        channel = len(np.shape(image))

        if self.auto_param:
            self.ratio = np.random.uniform(0, 0.001)
            self.k_x_range = (1, 0.1 * w)
            self.k_y_range = (1, 0.1 * h)

        for _ in range(5):
            if channel == 3:
                noise = np.ones((w, h, 3), dtype=np.uint8) * 255
                dst = np.ones((w, h, 3), dtype=np.uint8) * 255
            else:
                noise = np.ones((w, h), dtype=np.uint8) * 255
                dst = np.ones((w, h), dtype=np.uint8) * 255

            rate = self.ratio / 5
            mask = np.random.uniform(size=(w, h)) < rate
            noise[mask] = np.random.randint(0, randon_range)

            k_x, k_y = np.random.randint(*self.k_x_range), np.random.randint(*self.k_y_range)
            kernel = np.ones((k_x, k_y), np.uint8)
            erode = cv2.erode(noise, kernel, iterations=1)
            dst = erode * (erode < randon_range) + dst * (1 - erode < randon_range)
            # Add black point
            for _ in range(np.random.randint(math.ceil(k_x * k_y / 2))):
                x = np.random.randint(-k_x, k_x)
                y = np.random.randint(-k_y, k_y)
                matrix = np.array([[1, 0, y], [0, 1, x]], dtype=np.float)
                affine = cv2.warpAffine(noise, matrix, (h, w))
                dst = affine * (affine < randon_range) + dst * (1 - affine < randon_range)
            # Add white point
            for _ in range(int(k_x * k_y / 2)):
                x = np.random.randint(-k_x / 2 - 1, k_x / 2 + 1)
                y = np.random.randint(-k_y / 2 - 1, k_y / 2 + 1)
                matrix = np.array([[1, 0, y], [0, 1, x]], dtype=np.float)
                affine = cv2.warpAffine(noise, matrix, (h, w))
                white = affine < randon_range
                dst[white] = 255

        mask = dst < randon_range
        dst = image * (1 - mask) + dst * mask
        dst = self._original_format(dst, chw, normalized, gray3dim)

        return dst.astype(ori_dtype)

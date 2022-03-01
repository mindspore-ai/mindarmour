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
Image Blur
"""

import numpy as np
import cv2

from mindarmour.natural_robustness.transform.image.natural_perturb import _NaturalPerturb
from mindarmour.utils._check_param import check_param_multi_types, check_int_positive, check_param_type
from mindarmour.utils.logger import LogUtil

LOGGER = LogUtil.get_instance()
TAG = 'Image Blur'


class GaussianBlur(_NaturalPerturb):
    """
    Blurs the image using Gaussian blur filter.

    Args:
        ksize (int): Size of gaussian kernel, this value must be non-negnative.
        auto_param (bool): Auto selected parameters. Selected parameters will preserve semantics of image.

    Example:
        >>> img = cv2.imread('1.png')
        >>> img = np.array(img)
        >>> ksize = 5
        >>> trans = GaussianBlur(ksize)
        >>> dst = trans(img)
    """

    def __init__(self, ksize=2, auto_param=False):
        super(GaussianBlur, self).__init__()
        ksize = check_int_positive('ksize', ksize)
        if auto_param:
            ksize = 2 * np.random.randint(0, 5) + 1
        else:
            ksize = 2 * ksize + 1
        self.ksize = (ksize, ksize)

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
        new_img = cv2.GaussianBlur(image, self.ksize, 0)
        new_img = self._original_format(new_img, chw, normalized, gray3dim)
        return new_img.astype(ori_dtype)


class MotionBlur(_NaturalPerturb):
    """
    Motion blur for a given image.

    Args:
        degree (int): Degree of blur. This value must be positive. Suggested value range in [1, 15].
        angle: (union[float, int]): Direction of motion blur. Angle=0 means up and down motion blur. Angle is
            counterclockwise.
        auto_param (bool): Auto selected parameters. Selected parameters will preserve semantics of image.

    Example:
        >>> img = cv2.imread('1.png')
        >>> img = np.array(img)
        >>> angle = 0
        >>> degree = 5
        >>> trans = MotionBlur(degree=degree, angle=angle)
        >>> new_img = trans(img)
    """

    def __init__(self, degree=5, angle=45, auto_param=False):
        super(MotionBlur, self).__init__()
        self.degree = check_int_positive('degree', degree)
        self.degree = check_param_multi_types('degree', degree, [float, int])
        auto_param = check_param_type('auto_param', auto_param, bool)
        if auto_param:
            self.degree = np.random.randint(1, 5)
            self.angle = np.random.uniform(0, 360)
        else:
            self.angle = angle - 45

    def __call__(self, image):
        """
        Motion blur for a given image.

        Args:
            image (numpy.ndarray): Original image.

        Returns:
            numpy.ndarray, image after motion blur.
        """
        ori_dtype = image.dtype
        _, chw, normalized, gray3dim, image = self._check(image)
        matrix = cv2.getRotationMatrix2D((self.degree / 2, self.degree / 2), self.angle, 1)
        motion_blur_kernel = np.diag(np.ones(self.degree))
        motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, matrix, (self.degree, self.degree))
        motion_blur_kernel = motion_blur_kernel / self.degree
        blurred = cv2.filter2D(image, -1, motion_blur_kernel)
        # convert to uint8
        cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
        blurred = self._original_format(blurred, chw, normalized, gray3dim)

        return blurred.astype(ori_dtype)


class GradientBlur(_NaturalPerturb):
    """
    Gradient blur.

    Args:
        point (union[tuple, list]): 2D coordinate of the Blur center point.
        kernel_num (int): Number of blur kernels. Suggested value range in [1, 8].
        center (bool): Blurred or clear at the center of a specified point.
        auto_param (bool): Auto selected parameters. Selected parameters will preserve semantics of image.

    Example:
        >>> img = cv2.imread('xx.png')
        >>> img = np.array(img)
        >>> number = 5
        >>> h, w = img.shape[:2]
        >>> point = (int(h / 5), int(w / 5))
        >>> center = True
        >>> trans = GradientBlur(point, number,  center)
        >>> new_img = trans(img)
    """

    def __init__(self, point, kernel_num=3, center=True, auto_param=False):
        super(GradientBlur).__init__()
        point = check_param_multi_types('point', point, [list, tuple])
        self.auto_param = check_param_type('auto_param', auto_param, bool)
        self.point = tuple(point)
        self.kernel_num = check_int_positive('kernel_num', kernel_num)
        self.center = check_param_type('center', center, bool)

    def _auto_param(self, h, w):
        self.point = (int(np.random.uniform(0, h)), int(np.random.uniform(0, w)))
        self.kernel_num = np.random.randint(1, 6)
        self.center = np.random.choice([True, False])

    def __call__(self, image):
        """
        Args:
            image (numpy.ndarray): Original image.

        Returns:
            numpy.ndarray, gradient blurred image.
        """
        ori_dtype = image.dtype
        _, chw, normalized, gray3dim, image = self._check(image)
        w, h = image.shape[:2]
        if self.auto_param:
            self._auto_param(h, w)
        mask = np.zeros(image.shape, dtype=np.uint8)
        masks = []
        radius = max(w - self.point[0], self.point[0], h - self.point[1], self.point[1])
        radius = int(radius / self.kernel_num)
        for i in range(self.kernel_num):
            circle = cv2.circle(mask.copy(), self.point, radius * (1 + i), (1, 1, 1), -1)
            masks.append(circle)
        blurs = []
        for i in range(3, 3 + 2 * self.kernel_num, 2):
            ksize = (i, i)
            blur = cv2.GaussianBlur(image, ksize, 0)
            blurs.append(blur)

        dst = image.copy()
        if self.center:
            for i in range(self.kernel_num):
                dst = masks[i] * dst + (1 - masks[i]) * blurs[i]
        else:
            for i in range(self.kernel_num - 1, -1, -1):
                dst = masks[i] * blurs[self.kernel_num - 1 - i] + (1 - masks[i]) * dst
        dst = self._original_format(dst, chw, normalized, gray3dim)
        return dst.astype(ori_dtype)

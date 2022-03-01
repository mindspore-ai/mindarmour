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
Image transformation.
"""
import math
import numpy as np
import cv2

from mindarmour.natural_robustness.transform.image.natural_perturb import _NaturalPerturb
from mindarmour.utils._check_param import check_param_multi_types, check_param_type, check_value_non_negative
from mindarmour.utils.logger import LogUtil

LOGGER = LogUtil.get_instance()
TAG = 'Image Transformation'


class Translate(_NaturalPerturb):
    """
    Translate an image.

    Args:
        x_bias (Union[int, float]): X-direction translation, x = x + x_bias*image_width. Suggested value range
            in [-0.1, 0.1].
        y_bias (Union[int, float]): Y-direction translation,  y = y + y_bias*image_length. Suggested value range
            in [-0.1, 0.1].
        auto_param (bool): Auto selected parameters. Selected parameters will preserve semantics of image.

    Example:
        >>> img = cv2.imread('1.png')
        >>> img = np.array(img)
        >>> x_bias = 0.1
        >>> y_bias = 0.1
        >>> trans = Translate(x_bias, y_bias)
        >>> dst = trans(img)
    """

    def __init__(self, x_bias=0, y_bias=0, auto_param=False):
        super(Translate, self).__init__()
        self.x_bias = check_param_multi_types('x_bias', x_bias, [int, float])
        self.y_bias = check_param_multi_types('y_bias', y_bias, [int, float])
        if auto_param:
            self.x_bias = np.random.uniform(-0.1, 0.1)
            self.y_bias = np.random.uniform(-0.1, 0.1)

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
        h, w = image.shape[:2]
        matrix = np.array([[1, 0, self.x_bias * w], [0, 1, self.y_bias * h]], dtype=np.float)
        new_img = cv2.warpAffine(image, matrix, (w, h))
        new_img = self._original_format(new_img, chw, normalized, gray3dim)
        return new_img.astype(ori_dtype)


class Scale(_NaturalPerturb):
    """
    Scale an image in the middle.

    Args:
        factor_x (Union[float, int]): Rescale in X-direction, x=factor_x*x. Suggested value range in [0.5, 1] and
            abs(factor_y - factor_x) < 0.5.
        factor_y (Union[float, int]): Rescale in Y-direction, y=factor_y*y. Suggested value range in [0.5, 1] and
            abs(factor_y - factor_x) < 0.5.
        auto_param (bool): Auto selected parameters. Selected parameters will preserve semantics of image.

    Example:
        >>> img = cv2.imread('1.png')
        >>> img = np.array(img)
        >>> factor_x = 0.7
        >>> factor_y = 0.6
        >>> trans = Scale(factor_x, factor_y)
        >>> dst = trans(img)
    """

    def __init__(self, factor_x=1, factor_y=1, auto_param=False):
        super(Scale, self).__init__()
        self.factor_x = check_param_multi_types('factor_x', factor_x, [int, float])
        self.factor_y = check_param_multi_types('factor_y', factor_y, [int, float])
        auto_param = check_param_type('auto_param', auto_param, bool)
        if auto_param:
            self.factor_x = np.random.uniform(0.5, 1)
            self.factor_y = np.random.uniform(0.5, 1)

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
        h, w = image.shape[:2]
        matrix = np.array([[self.factor_x, 0, 0], [0, self.factor_y, 0]], dtype=np.float)
        new_img = cv2.warpAffine(image, matrix, (w, h))
        new_img = self._original_format(new_img, chw, normalized, gray3dim)
        return new_img.astype(ori_dtype)


class Shear(_NaturalPerturb):
    """
    Shear an image, for each pixel (x, y) in the sheared image, the new value is taken from a position
    (x+factor_x*y, factor_y*x+y) in the origin image. Then the sheared image will be rescaled to fit original size.

    Args:
        factor (Union[float, int]): Shear rate in shear direction. Suggested value range in [0.05, 0.5].
        direction (str): Direction of deformation. Optional value is 'vertical' or 'horizontal'.
        auto_param (bool): Auto selected parameters. Selected parameters will preserve semantics of image.

    Example:
        >>> img = cv2.imread('1.png')
        >>> img = np.array(img)
        >>> factor = 0.2
        >>> trans = Shear(factor, direction='horizontal')
        >>> dst = trans(img)
    """

    def __init__(self, factor=0.2, direction='horizontal', auto_param=False):
        super(Shear, self).__init__()
        self.factor = check_param_multi_types('factor', factor, [int, float])
        if direction not in ['horizontal', 'vertical']:
            msg = "'direction must be in ['horizontal', 'vertical'], but got {}".format(direction)
            raise ValueError(msg)
        self.direction = direction
        auto_param = check_param_type('auto_params', auto_param, bool)
        if auto_param:
            self.factor = np.random.uniform(0.05, 0.5)
            self.direction = np.random.choice(['horizontal', 'vertical'])

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
        h, w = image.shape[:2]
        if self.direction == 'horizontal':
            matrix = np.array([[1, self.factor, 0], [0, 1, 0]], dtype=np.float)
            nw = int(w + self.factor * h)
            nh = h
        else:
            matrix = np.array([[1, 0, 0], [self.factor, 1, 0]], dtype=np.float)
            nw = w
            nh = int(h + self.factor * w)
        new_img = cv2.warpAffine(image, matrix, (nw, nh))
        new_img = cv2.resize(new_img, (w, h))
        new_img = self._original_format(new_img, chw, normalized, gray3dim)
        return new_img.astype(ori_dtype)


class Rotate(_NaturalPerturb):
    """
    Rotate an image of counter clockwise around its center.

    Args:
        angle (Union[float, int]): Degrees of counter clockwise. Suggested value range in [-60, 60].
        auto_param (bool): Auto selected parameters. Selected parameters will preserve semantics of image.

    Example:
        >>> img = cv2.imread('1.png')
        >>> img = np.array(img)
        >>> angle = 20
        >>> trans = Rotate(angle)
        >>> dst = trans(img)
    """

    def __init__(self, angle=20, auto_param=False):
        super(Rotate, self).__init__()
        self.angle = check_param_multi_types('angle', angle, [int, float])
        auto_param = check_param_type('auto_param', auto_param, bool)
        if auto_param:
            self.angle = np.random.uniform(0, 360)

    def __call__(self, image):
        """
        Transform the image.

        Args:
            image (numpy.ndarray): Original image to be transformed.

        Returns:
            numpy.ndarray, rotated image.
        """
        ori_dtype = image.dtype
        _, chw, normalized, gray3dim, image = self._check(image)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, -self.angle, 1.0)
        cos = np.abs(matrix[0, 0])
        sin = np.abs(matrix[0, 1])

        # Calculate new edge after rotated
        nw = int((h * sin) + (w * cos))
        nh = int((h * cos) + (w * sin))
        # Adjust move distance of rotate matrix.
        matrix[0, 2] += (nw / 2) - center[0]
        matrix[1, 2] += (nh / 2) - center[1]
        rotate = cv2.warpAffine(image, matrix, (nw, nh))
        rotate = cv2.resize(rotate, (w, h))
        new_img = self._original_format(rotate, chw, normalized, gray3dim)
        return new_img.astype(ori_dtype)


class Perspective(_NaturalPerturb):
    """
    Perform perspective transformation on a given picture.

    Args:
        ori_pos (list): Four points in original image.
        dst_pos (list): The point coordinates of the 4 points in ori_pos after perspective transformation.
        auto_param (bool): Auto selected parameters. Selected parameters will preserve semantics of image.

    Example:
        >>> img = cv2.imread('1.png')
        >>> img = np.array(img)
        >>> ori_pos = [[0, 0], [0, 800], [800, 0], [800, 800]]
        >>> dst_pos = [[50, 0], [0, 800], [780, 0], [800, 800]]
        >>> trans = Perspective(ori_pos, dst_pos)
        >>> dst = trans(img)
    """

    def __init__(self, ori_pos, dst_pos, auto_param=False):
        super(Perspective, self).__init__()
        ori_pos = check_param_type('ori_pos', ori_pos, list)
        dst_pos = check_param_type('dst_pos', dst_pos, list)
        self.ori_pos = np.float32(ori_pos)
        self.dst_pos = np.float32(dst_pos)
        self.auto_param = check_param_type('auto_param', auto_param, bool)

    def _set_auto_param(self, w, h):
        self.ori_pos = [[h * 0.25, w * 0.25], [h * 0.25, w * 0.75], [h * 0.75, w * 0.25], [h * 0.75, w * 0.75]]
        self.dst_pos = [[np.random.uniform(0, h * 0.5), np.random.uniform(0, w * 0.5)],
                        [np.random.uniform(0, h * 0.5), np.random.uniform(w * 0.5, w)],
                        [np.random.uniform(h * 0.5, h), np.random.uniform(0, w * 0.5)],
                        [np.random.uniform(h * 0.5, h), np.random.uniform(w * 0.5, w)]]
        self.ori_pos = np.float32(self.ori_pos)
        self.dst_pos = np.float32(self.dst_pos)

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
        h, w = image.shape[:2]
        if self.auto_param:
            self._set_auto_param(w, h)
        matrix = cv2.getPerspectiveTransform(self.ori_pos, self.dst_pos)
        new_img = cv2.warpPerspective(image, matrix, (w, h))
        new_img = self._original_format(new_img, chw, normalized, gray3dim)
        return new_img.astype(ori_dtype)


class Curve(_NaturalPerturb):
    """
    Curve picture using sin method.

    Args:
        curves (union[float, int]): Divide width to curves of `2*math.pi`, which means how many curve cycles. Suggested
            value range in [0.1. 5].
        depth (union[float, int]): Amplitude of sin method. Suggested value not exceed 1/10 of the length of the
            picture.
        mode (str): Direction of deformation. Optional value is 'vertical' or 'horizontal'.
        auto_param (bool): Auto selected parameters. Selected parameters will preserve semantics of image.

    Examples:
        >>> img = cv2.imread('x.png')
        >>> curves =1
        >>> depth = 10
        >>> trans = Curve(curves, depth, mode='vertical')
        >>> img_new = trans(img)
    """

    def __init__(self, curves=3, depth=10, mode='vertical', auto_param=False):
        super(Curve).__init__()
        self.curves = check_value_non_negative('curves', curves)
        self.depth = check_value_non_negative('depth', depth)
        if mode in ['vertical', 'horizontal']:
            self.mode = mode
        else:
            msg = "Value of param mode must be in ['vertical', 'horizontal']"
            LOGGER.error(TAG, msg)
            raise ValueError(msg)
        self.auto_param = check_param_type('auto_param', auto_param, bool)

    def _set_auto_params(self, height, width):
        if self.auto_param:
            self.curves = np.random.uniform(1, 5)
            self.mode = np.random.choice(['vertical', 'horizontal'])
            if self.mode == 'vertical':
                self.depth = np.random.uniform(1, 0.1 * width)
            else:
                self.depth = np.random.uniform(1, 0.1 * height)

    def __call__(self, image):
        """
        Curve picture using sin method.

        Args:
            image (numpy.ndarray): Original image.

        Returns:
            numpy.ndarray, curved image.
        """
        ori_dtype = image.dtype
        _, chw, normalized, gray3dim, image = self._check(image)
        shape = image.shape
        height, width = shape[:2]
        if self.mode == 'vertical':
            if len(shape) == 3:
                image = np.transpose(image, [1, 0, 2])
            else:
                image = np.transpose(image, [1, 0])
        src_x = np.zeros((height, width), np.float32)
        src_y = np.zeros((height, width), np.float32)

        for y in range(height):
            for x in range(width):
                src_x[y, x] = x
                src_y[y, x] = y + self.depth * math.sin(x / (width / self.curves / 2 / math.pi))
        img_new = cv2.remap(image, src_x, src_y, cv2.INTER_LINEAR)

        if self.mode == 'vertical':
            if len(shape) == 3:
                img_new = np.transpose(img_new, [1, 0, 2])
            else:
                img_new = np.transpose(image, [1, 0])
        new_img = self._original_format(img_new, chw, normalized, gray3dim)
        return new_img.astype(ori_dtype)

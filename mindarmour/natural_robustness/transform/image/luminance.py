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
Image luminance.
"""
import math
import numpy as np
import cv2

from mindarmour.natural_robustness.transform.image.natural_perturb import _NaturalPerturb
from mindarmour.utils._check_param import check_param_multi_types, check_param_in_range, check_param_type, \
    check_value_non_negative
from mindarmour.utils.logger import LogUtil

LOGGER = LogUtil.get_instance()
TAG = 'Image Luminance'


class Contrast(_NaturalPerturb):
    """
    Contrast of an image.

    Args:
        alpha (Union[float, int]): Control the contrast of an image. :math:`out_image = in_image*alpha+beta`.
            Suggested value range in [0.2, 2].
        beta (Union[float, int]): Delta added to alpha. Default: 0.
        auto_param (bool): Auto selected parameters. Selected parameters will preserve semantics of image.

    Example:
        >>> img = cv2.imread('1.png')
        >>> img = np.array(img)
        >>> alpha = 0.1
        >>> beta = 1
        >>> trans = Contrast(alpha, beta)
        >>> dst = trans(img)
    """

    def __init__(self, alpha=1, beta=0, auto_param=False):
        super(Contrast, self).__init__()
        self.alpha = check_param_multi_types('factor', alpha, [int, float])
        self.beta = check_param_multi_types('factor', beta, [int, float])
        auto_param = check_param_type('auto_param', auto_param, bool)
        if auto_param:
            self.alpha = np.random.uniform(0.2, 2)
            self.beta = np.random.uniform(-20, 20)

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
        dst = cv2.convertScaleAbs(image, alpha=self.alpha, beta=self.beta)
        dst = self._original_format(dst, chw, normalized, gray3dim)
        return dst.astype(ori_dtype)


def _circle_gradient_mask(img_src, color_start, color_end, scope=0.5, point=None):
    """
    Generate circle gradient mask.

    Args:
        img_src (numpy.ndarray): Source image.
        color_start (union([tuple, list])): Color of circle gradient center.
        color_end (union([tuple, list])): Color of circle gradient edge.
        scope (float): Range of the gradient. A larger value indicates a larger gradient range.
        point (union([tuple, list]): Gradient center point.

    Returns:
        numpy.ndarray, gradients mask.
    """
    if not isinstance(img_src, np.ndarray):
        raise TypeError('`src` must be numpy.ndarray type, but got {0}.'.format(type(img_src)))

    shape = img_src.shape
    height, width = shape[:2]
    rgb = False
    if len(shape) == 3:
        rgb = True
    if point is None:
        point = (height // 2, width // 2)
    x, y = point

    # upper left
    bound_upper_left = math.ceil(math.sqrt(x ** 2 + y ** 2))
    # upper right
    bound_upper_right = math.ceil(math.sqrt(height ** 2 + (width - y) ** 2))
    # lower left
    bound_lower_left = math.ceil(math.sqrt((height - x) ** 2 + y ** 2))
    # lower right
    bound_lower_right = math.ceil(math.sqrt((height - x) ** 2 + (width - y) ** 2))

    radius = max(bound_lower_left, bound_lower_right, bound_upper_left, bound_upper_right) * scope

    img_grad = np.ones_like(img_src, dtype=np.uint8) * max(color_end)
    # opencv use BGR format
    grad_b = float(color_end[0] - color_start[0]) / radius
    grad_g = float(color_end[1] - color_start[1]) / radius
    grad_r = float(color_end[2] - color_start[2]) / radius

    for i in range(height):
        for j in range(width):
            distance = math.ceil(math.sqrt((x - i) ** 2 + (y - j) ** 2))
            if distance >= radius:
                continue
            if rgb:
                img_grad[i, j, 0] = color_start[0] + distance * grad_b
                img_grad[i, j, 1] = color_start[1] + distance * grad_g
                img_grad[i, j, 2] = color_start[2] + distance * grad_r
            else:
                img_grad[i, j] = color_start[0] + distance * grad_b

    return img_grad.astype(np.uint8)


def _line_gradient_mask(image, start_pos=None, start_color=(0, 0, 0), end_color=(255, 255, 255), mode='horizontal'):
    """
    Generate liner gradient mask.

    Args:
        image (numpy.ndarray): Original image.
        start_pos (union[tuple, list]): 2D coordinate of gradient center.
        start_color (union([tuple, list])): Color of circle gradient center.
        end_color (union([tuple, list])): Color of circle gradient edge.
        mode　(str): Direction of gradient. Optional value is 'vertical' or 'horizontal'.

    Returns:
        numpy.ndarray, gradients mask.
    """
    shape = image.shape
    h, w = shape[:2]
    rgb = False
    if len(shape) == 3:
        rgb = True
    if start_pos is None:
        start_pos = 0.5
    else:
        if mode == 'horizontal':
            if start_pos[0] > h:
                start_pos = 1
            else:
                start_pos = start_pos[0] / h
        else:
            if start_pos[1] > w:
                start_pos = 1
            else:
                start_pos = start_pos[1] / w
    start_color = np.array(start_color)
    end_color = np.array(end_color)
    if mode == 'horizontal':
        w_l = int(w * start_pos)
        w_r = w - w_l
        if w_l > w_r:
            r_end_color = (end_color - start_color) / start_pos * (1 - start_pos) + start_color
            left = np.linspace(end_color, start_color, w_l)
            right = np.linspace(start_color, r_end_color, w_r)
        else:
            l_end_color = (end_color - start_color) / (1 - start_pos) * start_pos + start_color
            left = np.linspace(l_end_color, start_color, w_l)
            right = np.linspace(start_color, end_color, w_r)
        line = np.concatenate((left, right), axis=0)
        mask = np.reshape(np.tile(line, (h, 1)), (h, w, 3))
    else:
        # 'vertical'
        h_t = int(h * start_pos)
        h_b = h - h_t
        if h_t > h_b:
            b_end_color = (end_color - start_color) / start_pos * (1 - start_pos) + start_color
            top = np.linspace(end_color, start_color, h_t)
            bottom = np.linspace(start_color, b_end_color, h_b)
        else:
            t_end_color = (end_color - start_color) / (1 - start_pos) * start_pos + start_color
            top = np.linspace(t_end_color, start_color, h_t)
            bottom = np.linspace(start_color, end_color, h_b)
        line = np.concatenate((top, bottom), axis=0)
        mask = np.reshape(np.tile(line, (w, 1)), (w, h, 3))
        mask = np.transpose(mask, [1, 0, 2])
    if not rgb:
        mask = mask[:, :, 0]
    return mask.astype(np.uint8)


class GradientLuminance(_NaturalPerturb):
    """
    Gradient adjusts the luminance of picture.

    Args:
        color_start (union[tuple, list]): Color of gradient center. Default:(0, 0, 0).
        color_end (union[tuple, list]):  Color of gradient edge. Default:(255, 255, 255).
        start_point (union[tuple, list]): 2D coordinate of gradient center.
        scope (float): Range of the gradient. A larger value indicates a larger gradient range. Default: 0.3.
        pattern (str): Dark or light, this value must be in ['light', 'dark'].
        bright_rate (float): Control brightness. A larger value indicates a larger gradient range. If parameter
            'pattern' is 'light', Suggested value range in [0.1, 0.7], if parameter 'pattern' is 'dark', Suggested value
            range in [0.1, 0.9].
        mode (str): Gradient mode, value must be in ['circle', 'horizontal', 'vertical'].
        auto_param (bool): Auto selected parameters. Selected parameters will preserve semantics of image.

    Examples:
        >>> img = cv2.imread('x.png')
        >>> height, width = img.shape[:2]
        >>> point = (height // 4, width // 2)
        >>> start = (255, 255, 255)
        >>> end = (0, 0, 0)
        >>> scope = 0.3
        >>> pattern='light'
        >>> bright_rate = 0.3
        >>> trans = GradientLuminance(start, end, point, scope, pattern, bright_rate, mode='circle')
        >>> img_new = trans(img)
    """

    def __init__(self, color_start=(0, 0, 0), color_end=(255, 255, 255), start_point=(10, 10), scope=0.5,
                 pattern='light', bright_rate=0.3, mode='circle', auto_param=False):
        super(GradientLuminance, self).__init__()
        self.color_start = check_param_multi_types('color_start', color_start, [list, tuple])
        self.color_end = check_param_multi_types('color_end', color_end, [list, tuple])
        self.start_point = check_param_multi_types('start_point', start_point, [list, tuple])
        self.scope = check_value_non_negative('scope', scope)
        self.bright_rate = check_param_type('bright_rate', bright_rate, float)
        self.bright_rate = check_param_in_range('bright_rate', bright_rate, 0, 1)
        self.auto_param = check_param_type('auto_param', auto_param, bool)

        if pattern in ['light', 'dark']:
            self.pattern = pattern
        else:
            msg = "Value of param pattern must be in ['light', 'dark']"
            LOGGER.error(TAG, msg)
            raise ValueError(msg)
        if mode in ['circle', 'horizontal', 'vertical']:
            self.mode = mode
        else:
            msg = "Value of param mode must be in ['circle', 'horizontal', 'vertical']"
            LOGGER.error(TAG, msg)
            raise ValueError(msg)

    def _set_auto_param(self, w, h):
        self.color_start = (np.random.uniform(0, 255),) * 3
        self.color_end = (np.random.uniform(0, 255),) * 3
        self.start_point = (np.random.uniform(0, w), np.random.uniform(0, h))
        self.scope = np.random.uniform(0, 1)
        self.bright_rate = np.random.uniform(0.1, 0.9)
        self.pattern = np.random.choice(['light', 'dark'])
        self.mode = np.random.choice(['circle', 'horizontal', 'vertical'])

    def __call__(self, image):
        """
        Gradient adjusts the luminance of picture.

        Args:
            image (numpy.ndarray): Original image.

        Returns:
            numpy.ndarray, image with perlin noise.
        """
        ori_dtype = image.dtype
        _, chw, normalized, gray3dim, image = self._check(image)
        w, h = image.shape[:2]
        if self.auto_param:
            self._set_auto_param(w, h)
        if self.mode == 'circle':
            mask = _circle_gradient_mask(image, self.color_start, self.color_end, self.scope, self.start_point)
        else:
            mask = _line_gradient_mask(image, self.start_point, self.color_start, self.color_end, mode=self.mode)

        if self.pattern == 'light':
            img_new = cv2.addWeighted(image, 1, mask, self.bright_rate, 0.0)
        else:
            img_new = cv2.addWeighted(image, self.bright_rate, mask, 1 - self.bright_rate, 0.0)
        img_new = self._original_format(img_new, chw, normalized, gray3dim)
        return img_new.astype(ori_dtype)

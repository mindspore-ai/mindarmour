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
Image transform
"""
import os
import math
import numpy as np
import cv2
from perlin_numpy import generate_fractal_noise_2d

from mindarmour.utils._check_param import check_param_multi_types, check_param_in_range, check_numpy_param, \
    check_int_positive, check_param_type, check_value_non_negative
from mindarmour.utils.logger import LogUtil

LOGGER = LogUtil.get_instance()
TAG = 'Image Transformation'


class Contrast:
    """
    Contrast of an image.

    Args:
        alpha (Union[float, int]): Control the contrast of an image. Suggested value range in [0.2, 2].
        beta (Union[float, int]): Delta added to alpha. Default: 0.

    Example:
        >>> img = cv2.imread('1.png')
        >>> img = np.array(img)
        >>> alpha = 0.1
        >>> beta = 1
        >>> trans = Contrast(alpha, beta)
        >>> dst = trans(img)
    """

    def __init__(self, alpha=1, beta=0):
        super(Contrast, self).__init__()
        self.alpha = check_param_multi_types('factor', alpha, [int, float])
        self.beta = check_param_multi_types('factor', beta, [int, float])

    def __call__(self, image):
        """
        Transform the image.

        Args:
            image (numpy.ndarray): Original image to be transformed.

        Returns:
            numpy.ndarray, transformed image.
        """
        image = check_numpy_param('image', image)
        new_img = cv2.convertScaleAbs(image, alpha=self.alpha, beta=self.beta)
        return new_img


class GaussianBlur:
    """
    Blurs the image using Gaussian blur filter.

    Args:
        ksize (Union[list, tuple]): Size of  gaussian kernel.

    Example:
        >>> img = cv2.imread('1.png')
        >>> img = np.array(img)
        >>> ksize = 0.1
        >>> trans = GaussianBlur(ksize)
        >>> dst = trans(img)
    """

    def __init__(self, ksize=5):
        super(GaussianBlur, self).__init__()
        ksize = check_int_positive('ksize', ksize)
        self.ksize = (ksize, ksize)

    def __call__(self, image):
        """
        Transform the image.

        Args:
            image (numpy.ndarray): Original image to be transformed.

        Returns:
            numpy.ndarray, transformed image.
        """
        image = check_numpy_param('image', image)
        new_img = cv2.GaussianBlur(image, self.ksize, 0)
        return new_img


class SaltAndPepperNoise:
    """
    Add noise of an image.

    Args:
        factor (float): Noise density, the proportion of noise points per unit pixel area. Suggested value range in
            [0.001, 0.15].

    Example:
        >>> img = cv2.imread('1.png')
        >>> img = np.array(img)
        >>> factor = 0.1
        >>> trans = SaltAndPepperNoise(factor)
        >>> dst = trans(img)
    """

    def __init__(self, factor=0):
        super(SaltAndPepperNoise, self).__init__()
        self.factor = check_param_multi_types('factor', factor, [int, float])

    def __call__(self, image):
        """
        Transform the image.

        Args:
            image (numpy.ndarray): Original image to be transformed.

        Returns:
            numpy.ndarray, transformed image.
        """
        image = check_numpy_param('image', image)
        ori_dtype = image.dtype
        noise = np.random.uniform(low=-1, high=1, size=(image.shape[0], image.shape[1]))
        trans_image = np.copy(image)
        threshold = 1 - self.factor
        trans_image[noise < -threshold] = (0, 0, 0)
        trans_image[noise > threshold] = (255, 255, 255)
        return trans_image.astype(ori_dtype)


class Translate:
    """
    Translate an image.

    Args:
        x_bias (Union[int, float]): X-direction translation, x = x + x_bias*image_length. Suggested value range
            in [-0.1, 0.1].
        y_bias (Union[int, float]): Y-direction translation,  y = y + y_bias*image_wide. Suggested value range
            in [-0.1, 0.1].

    Example:
        >>> img = cv2.imread('1.png')
        >>> img = np.array(img)
        >>> x_bias = 0.1
        >>> y_bias = 0.1
        >>> trans = Translate(x_bias, y_bias)
        >>> dst = trans(img)
    """

    def __init__(self, x_bias=0, y_bias=0):
        super(Translate, self).__init__()
        self.x_bias = check_param_multi_types('x_bias', x_bias, [int, float])
        self.y_bias = check_param_multi_types('y_bias', y_bias, [int, float])

    def __call__(self, image):
        """
        Transform the image.

        Args:
            image (numpy.ndarray): Original image to be transformed.

        Returns:
            numpy.ndarray, transformed image.
        """
        image = check_numpy_param('image', image)
        h, w = image.shape[:2]
        matrix = np.array([[1, 0, self.x_bias * w], [0, 1, self.y_bias * h]], dtype=np.float)
        new_img = cv2.warpAffine(image, matrix, (w, h))
        return new_img


class Scale:
    """
    Scale an image in the middle.

    Args:
        factor_x (Union[float, int]): Rescale in X-direction, x=factor_x*x. Suggested value range in [0.5, 1] and
            abs(factor_y - factor_x) < 0.5.
        factor_y (Union[float, int]): Rescale in Y-direction, y=factor_y*y. Suggested value range in [0.5, 1] and
            abs(factor_y - factor_x) < 0.5.

    Example:
        >>> img = cv2.imread('1.png')
        >>> img = np.array(img)
        >>> factor_x = 0.7
        >>> factor_y = 0.6
        >>> trans = Scale(factor_x, factor_y)
        >>> dst = trans(img)
    """

    def __init__(self, factor_x=1, factor_y=1):
        super(Scale, self).__init__()
        self.factor_x = check_param_multi_types('factor_x', factor_x, [int, float])
        self.factor_y = check_param_multi_types('factor_y', factor_y, [int, float])

    def __call__(self, image):
        """
        Transform the image.

        Args:
            image (numpy.ndarray): Original image to be transformed.

        Returns:
            numpy.ndarray, transformed image.
        """
        image = check_numpy_param('image', image)
        h, w = image.shape[:2]
        matrix = np.array([[self.factor_x, 0, 0], [0, self.factor_y, 0]], dtype=np.float)
        new_img = cv2.warpAffine(image, matrix, (w, h))
        return new_img


class Shear:
    """
    Shear an image, for each pixel (x, y) in the sheared image, the new value is taken from a position
    (x+factor_x*y, factor_y*x+y) in the origin image. Then the sheared image will be rescaled to fit original size.

    Args:
        factor (Union[float, int]): Shear rate in shear direction. Suggested value range in [0.05, 0.5].
        direction (str): Direction of deformation. Optional value is 'vertical' or 'horizontal'.

    Example:
        >>> img = cv2.imread('1.png')
        >>> img = np.array(img)
        >>> factor = 0.2
        >>> trans = Shear(factor, direction='horizontal')
        >>> dst = trans(img)
    """

    def __init__(self, factor, direction='horizontal'):
        super(Shear, self).__init__()
        self.factor = check_param_multi_types('factor', factor, [int, float])
        if direction not in ['horizontal', 'vertical']:
            msg = "'direction must be in ['horizontal', 'vertical'], but got {}".format(direction)
            raise ValueError(msg)
        self.direction = direction

    def __call__(self, image):
        """
        Transform the image.

        Args:
            image (numpy.ndarray): Original image to be transformed.

        Returns:
            numpy.ndarray, transformed image.
        """
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
        return new_img


class Rotate:
    """
    Rotate an image of counter clockwise around its center.

    Args:
        angle (Union[float, int]): Degrees of counter clockwise. Suggested value range in [-60, 60].

    Example:
        >>> img = cv2.imread('1.png')
        >>> img = np.array(img)
        >>> angle = 20
        >>> trans = Rotate(angle)
        >>> dst = trans(img)
    """
    def __init__(self, angle=20):
        super(Rotate, self).__init__()
        self.angle = check_param_multi_types('angle', angle, [int, float])

    def __call__(self, image):
        """
        Transform the image.

        Args:
            image (numpy.ndarray): Original image to be transformed.

        Returns:
            numpy.ndarray, rotated image.
        """
        image = check_numpy_param('image', image)
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
        return rotate


class Perspective:
    """
    Perform perspective transformation on a given picture.

    Args:
        ori_pos (list): Four points in original image.
        dst_pos (list): The point coordinates of the 4 points in ori_pos after perspective transformation.

    Example:
        >>> img = cv2.imread('1.png')
        >>> img = np.array(img)
        >>> ori_pos = [[0, 0], [0, 800], [800, 0], [800, 800]]
        >>> dst_pos = [[50, 0], [0, 800], [780, 0], [800, 800]]
        >>> trans = Perspective(ori_pos, dst_pos)
        >>> dst = trans(img)
    """

    def __init__(self, ori_pos, dst_pos):
        super(Perspective, self).__init__()
        ori_pos = check_param_type('ori_pos', ori_pos, list)
        dst_pos = check_param_type('dst_pos', dst_pos, list)
        self.ori_pos = np.float32(ori_pos)
        self.dst_pos = np.float32(dst_pos)

    def __call__(self, image):
        """
        Transform the image.

        Args:
            image (numpy.ndarray): Original image to be transformed.

        Returns:
            numpy.ndarray, transformed image.
        """
        image = check_numpy_param('image', image)
        h, w = image.shape[:2]
        matrix = cv2.getPerspectiveTransform(self.ori_pos, self.dst_pos)
        new_img = cv2.warpPerspective(image, matrix, (w, h))
        return new_img


class MotionBlur:
    """
    Motion blur for a given image.

    Args:
        degree (int): Degree of blur. This value must be positive. Suggested value range in [1, 15].
        angle: (union[float, int]): Direction of motion blur. Angle=0 means up and down motion blur. Angle is
            counterclockwise.

    Example:
        >>> img = cv2.imread('1.png')
        >>> img = np.array(img)
        >>> angle = 0
        >>> degree = 5
        >>> trans = MotionBlur(degree=degree, angle=angle)
        >>> new_img = trans(img)
    """

    def __init__(self, degree=5, angle=45):
        super(MotionBlur, self).__init__()
        self.degree = check_int_positive('degree', degree)
        self.degree = check_param_multi_types('degree', degree, [float, int])
        self.angle = angle - 45

    def __call__(self, image):
        """
        Motion blur for a given image.

        Args:
            image (numpy.ndarray): Original image.

        Returns:
            numpy.ndarray, image after motion blur.
        """
        image = check_numpy_param('image', image)
        matrix = cv2.getRotationMatrix2D((self.degree / 2, self.degree / 2), self.angle, 1)
        motion_blur_kernel = np.diag(np.ones(self.degree))
        motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, matrix, (self.degree, self.degree))
        motion_blur_kernel = motion_blur_kernel / self.degree
        blurred = cv2.filter2D(image, -1, motion_blur_kernel)
        # convert to uint8
        cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
        blurred = np.array(blurred, dtype=np.uint8)
        return blurred


class GradientBlur:
    """
    Gradient blur.

    Args:
        point (union[tuple, list]): 2D coordinate of the Blur center point.
        kernel_num (int): Number of blur kernels. Suggested value range in [1, 8].
        center (bool): Blurred or clear at the center of a specified point.

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

    def __init__(self, point, kernel_num=3, center=True):
        super(GradientBlur).__init__()
        point = check_param_multi_types('point', point, [list, tuple])
        self.point = tuple(point)
        self.kernel_num = check_int_positive('kernel_num', kernel_num)
        self.center = check_param_type('center', center, bool)

    def __call__(self, image):
        """

        Args:
            image (numpy.ndarray): Original image.

        Returns:
            numpy.ndarray, gradient blurred image.
        """
        image = check_numpy_param('image', image)
        w, h = image.shape[:2]
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
        return dst


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

    height, width = img_src.shape[:2]

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
            img_grad[i, j, 0] = color_start[0] + distance * grad_b
            img_grad[i, j, 1] = color_start[1] + distance * grad_g
            img_grad[i, j, 2] = color_start[2] + distance * grad_r

    return img_grad


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
    h, w = image.shape[:2]
    if start_pos is None:
        start_pos = 0.5
    else:
        if mode == 'horizontal':
            start_pos = start_pos[0] / h
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
        mask = np.array(mask, dtype=np.uint8)
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
        mask = np.array(mask, dtype=np.uint8)
    return mask


class GradientLuminance:
    """
    Gradient adjusts the luminance of picture.

    Args:
        color_start (union[tuple, list]): Color of gradient center. Default:(0, 0, 0).
        color_end (union[tuple, list]):  Color of gradient edge. Default:(255, 255, 255).
        start_point (union[tuple, list]): 2D coordinate of gradient center.
        scope (float): Range of the gradient. A larger value indicates a larger gradient range. Default: 0.3.
        pattern (str): Dark or light, this value must be in ['light', 'dark'].
        bright_rate (float): Control brightness of . A larger value indicates a larger gradient range. If parameter
            'pattern' is 'light', Suggested value range in [0.1, 0.7], if parameter 'pattern' is 'dark', Suggested value
            range in [0.1, 0.9].
        mode (str): Gradient mode, value must be in ['circle', 'horizontal', 'vertical'].

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

    def __init__(self, color_start, color_end, start_point, scope=0.5, pattern='light', bright_rate=0.3, mode='circle'):
        self.color_start = check_param_multi_types('color_start', color_start, [list, tuple])
        self.color_end = check_param_multi_types('color_end', color_end, [list, tuple])
        self.start_point = check_param_multi_types('start_point', start_point, [list, tuple])
        self.scope = check_value_non_negative('scope', scope)
        self.bright_rate = check_param_type('bright_rate', bright_rate, float)
        self.bright_rate = check_param_in_range('bright_rate', bright_rate, 0, 1)

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

    def __call__(self, image):
        """
        Gradient adjusts the luminance of picture.

        Args:
            image (numpy.ndarray): Original image.

        Returns:
            numpy.ndarray, image with perlin noise.
        """
        image = check_numpy_param('image', image)
        if self.mode == 'circle':
            mask = _circle_gradient_mask(image, self.color_start, self.color_end, self.scope, self.start_point)
        else:
            mask = _line_gradient_mask(image, self.start_point, self.color_start, self.color_end, mode=self.mode)

        if self.pattern == 'light':
            img_new = cv2.addWeighted(image, 1, mask, self.bright_rate, 0.0)
        else:
            img_new = cv2.addWeighted(image, self.bright_rate, mask, 1 - self.bright_rate, 0.0)
        return img_new


class Perlin:
    """
    Add perlin noise to given image.

    Args:
        ratio (float): Noise density. Suggested value range in [0.05, 0.9].
        shade (float): The degree of background shade color. Suggested value range in [0.1, 0.5].

    Examples:
        >>> img = cv2.imread('xx.png')
        >>> img = np.array(img)
        >>> ratio = 0.2
        >>> shade = 0.1
        >>> trans = Perlin(ratio, shade)
        >>> new_img = trans(img)
    """

    def __init__(self, ratio, shade=0.1):
        super(Perlin).__init__()
        ratio = check_param_type('ratio', ratio, float)
        ratio = check_param_in_range('ratio', ratio, 0, 1)
        if ratio > 0.7:
            self.ratio = 7
        else:
            self.ratio = int(ratio * 10)
        shade = check_param_type('shade', shade, float)
        self.shade = check_param_in_range('shade', shade, 0, 1)

    def __call__(self, image):
        """
        Add perlin noise to given image.

        Args:
            image (numpy.ndarray): Original image.

        Returns:
            numpy.ndarray, image with perlin noise.
        """
        image = check_numpy_param('image', image)
        noise = generate_fractal_noise_2d((1024, 1024), (2 ** self.ratio, 2 ** self.ratio), 4)
        noise[noise < 0] = 0
        noise[noise > 1] = 1
        back = np.array((1 - noise) * 255, dtype=np.uint8)
        back = cv2.resize(back, (image.shape[1], image.shape[0]))
        back = np.resize(np.repeat(back, 3), image.shape)
        dst = cv2.addWeighted(image, 1 - self.shade, back, self.shade, 0)
        return dst


class BackShadow:
    """
    Add background picture to given image.

    Args:
        template_path (str): Path of template pictures file.
        shade (float): The weight of background. Suggested value range in [0.1, 0.7].

    Examples:
        >>> img = cv2.imread('xx.png')
        >>> img = np.array(img)
        >>> template_path = 'template/leaf'
        >>> shade = 0.2
        >>> trans = BackShadow(template_path, shade=shade)
        >>> new_img = trans(img)
    """

    def __init__(self, template_path, shade=0.1):
        super(BackShadow).__init__()
        if os.path.exists(template_path):
            self.template_path = template_path
        else:
            msg = "Template_path is not exist"
            LOGGER.error(TAG, msg)
            raise ValueError(msg)
        shade = check_param_type('shade', shade, float)
        self.shade = check_param_in_range('shade', shade, 0, 1)

    def __call__(self, image):
        """
        Add background picture to given image.

        Args:
            image (numpy.ndarray): Original image.

        Returns:
            numpy.ndarray, image with background shadow.
        """
        image = check_numpy_param('image', image)
        file = os.listdir(self.template_path)
        file_path = os.path.join(self.template_path, np.random.choice(file))
        shadow = cv2.imread(file_path)
        shadow = cv2.resize(shadow, (image.shape[1], image.shape[0]))
        dst = cv2.addWeighted(image, 1 - self.shade, shadow, self.shade, beta=0)
        return dst


class NaturalNoise:
    """
    Add natural noise to an image.

    Args:
        ratio (float): Noise density, the proportion of noise blocks per unit pixel area. Suggested value range in
            [0.00001, 0.001].
        k_x_range (union[list, tuple]): Value range of the noise block length.
        k_y_range (union[list, tuple]): Value range of the noise block width.

    Examples:
        >>> img = cv2.imread('xx.png')
        >>> img = np.array(img)
        >>> ratio = 0.0002
        >>> k_x_range = (1, 5)
        >>> k_y_range = (3, 25)
        >>> trans = NaturalNoise(ratio, k_x_range, k_y_range)
        >>> new_img = trans(img)
    """

    def __init__(self, ratio=0.0002, k_x_range=(1, 5), k_y_range=(3, 25)):
        super(NaturalNoise).__init__()
        self.ratio = check_param_type('ratio', ratio, float)
        k_x_range = check_param_multi_types('k_x_range', k_x_range, [list, tuple])
        k_y_range = check_param_multi_types('k_y_range', k_y_range, [list, tuple])
        self.k_x_range = tuple(k_x_range)
        self.k_y_range = tuple(k_y_range)

    def __call__(self, image):
        """
        Add natural noise to given image.

        Args:
            image (numpy.ndarray): Original image.

        Returns:
            numpy.ndarray, image with natural noise.
        """
        image = check_numpy_param('image', image)
        randon_range = 100
        w, h = image.shape[:2]
        dst = np.ones((w, h, 3), dtype=np.uint8) * 255
        for _ in range(5):
            noise = np.ones((w, h, 3), dtype=np.uint8) * 255
            rate = self.ratio / 5
            mask = np.random.uniform(size=(w, h)) < rate
            noise[mask] = np.random.randint(0, randon_range)

            k_x, k_y = np.random.randint(*self.k_x_range), np.random.randint(*self.k_y_range)
            kernel = np.ones((k_x, k_y), np.uint8)
            erode = cv2.erode(noise, kernel, iterations=1)
            dst = erode * (erode < randon_range) + dst * (1 - erode < randon_range)
            # Add black point
            for _ in range(np.random.randint(k_x * k_y / 2)):
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
        dst = np.array(dst, dtype=np.uint8)
        return dst


class Curve:
    """
    Curve picture using sin method.

    Args:
        curves (union[float, int]): Divide width to curves of `2*math.pi`, which means how many curve cycles.　Suggested
            value range in [0.1. 5].
        depth (union[float, int]): Amplitude of sin method. Suggested value not exceed 1/10 of the length of the picture.
        mode (str): Direction of deformation. Optional value is 'vertical' or 'horizontal'.

    Examples:
        >>> img = cv2.imread('x.png')
        >>> curves =1
        >>> depth = 10
        >>> trans = Curve(curves, depth, mode='vertical')
        >>> img_new = trans(img)
    """

    def __init__(self, curves=10, depth=10, mode='vertical'):
        super(Curve).__init__()
        self.curves = check_value_non_negative('curves', curves)
        self.depth = check_value_non_negative('depth', depth)
        if mode in ['vertical', 'horizontal']:
            self.mode = mode
        else:
            msg = "Value of param mode must be in ['vertical', 'horizontal']"
            LOGGER.error(TAG, msg)
            raise ValueError(msg)

    def __call__(self, image):
        """
        Curve picture using sin method.

        Args:
            image (numpy.ndarray): Original image.

        Returns:
            numpy.ndarray, curved image.
        """
        image = check_numpy_param('image', image)
        if self.mode == 'vertical':
            image = np.transpose(image, [1, 0, 2])
        heights, widths = image.shape[:2]

        src_x = np.zeros((heights, widths), np.float32)
        src_y = np.zeros((heights, widths), np.float32)

        for y in range(heights):
            for x in range(widths):
                src_x[y, x] = x
                src_y[y, x] = y + self.depth * math.sin(x / (widths / self.curves / 2 / math.pi))
        img_new = cv2.remap(image, src_x, src_y, cv2.INTER_LINEAR)
        if self.mode == 'vertical':
            img_new = np.transpose(img_new, [1, 0, 2])
        return img_new


class BackgroundWord:
    """
    Overlay the background image on the original image.

    Args:
        shade (float): The weight of background. Suggested value range in [0.05, 0.3].
        back (numpy.ndarray): Background Image. If none, mean background image is the as original image.

    Examples:
        >>> img = cv2.imread('x.png')
        >>> back = cv2.imread('x.png')
        >>> shade=0.2
        >>> trans = BackgroundWord(shade, back)
        >>> img_new = trans(img)
    """

    def __init__(self, shade=0.1, back=None):
        super(BackgroundWord).__init__()
        self.shade = shade
        self.back = back

    def __call__(self, image):
        """
        Overlay the background image on the original image.

        Args:
            image (numpy.ndarray): Original image.

        Returns:
            numpy.ndarray, curved image.
        """
        image = check_numpy_param('image', image)
        beta = 0
        width, height = image.shape[:2]
        x = np.random.randint(0, int(width / 5))
        y = np.random.randint(0, int(height / 5))
        matrix = np.array([[1, 0, y], [0, 1, x]], dtype=np.float)
        affine = cv2.warpAffine(image.copy(), matrix, (height, width))
        back = image.copy()
        back[x:, y:] = affine[x:, y:]
        dst = cv2.addWeighted(image, 1 - self.shade, back, self.shade, beta)
        return dst

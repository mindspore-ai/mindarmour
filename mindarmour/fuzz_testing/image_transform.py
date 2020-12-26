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
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from mindspore.dataset.vision.py_transforms_util import is_numpy, \
    to_pil, hwc_to_chw
from mindarmour.utils._check_param import check_param_multi_types, check_param_in_range, check_numpy_param
from mindarmour.utils.logger import LogUtil

LOGGER = LogUtil.get_instance()
TAG = 'Image Transformation'


def chw_to_hwc(img):
    """
    Transpose the input image; shape (C, H, W) to shape (H, W, C).

    Args:
        img (numpy.ndarray): Image to be converted.

    Returns:
        img (numpy.ndarray), Converted image.
    """
    if is_numpy(img):
        return img.transpose(1, 2, 0).copy()
    raise TypeError('img should be numpy.ndarray. Got {}'.format(type(img)))


def is_hwc(img):
    """
    Check if the input image is shape (H, W, C).

    Args:
        img (numpy.ndarray): Image to be checked.

    Returns:
        Bool, True if input is shape (H, W, C).
    """
    if is_numpy(img):
        img_shape = np.shape(img)
        if img_shape[2] == 3 and img_shape[1] > 3 and img_shape[0] > 3:
            return True
        return False
    raise TypeError('img should be numpy.ndarray. Got {}'.format(type(img)))


def is_chw(img):
    """
    Check if the input image is shape (H, W, C).

    Args:
        img (numpy.ndarray): Image to be checked.

    Returns:
        Bool, True if input is shape (H, W, C).
    """
    if is_numpy(img):
        img_shape = np.shape(img)
        if img_shape[0] == 3 and img_shape[1] > 3 and img_shape[2] > 3:
            return True
        return False
    raise TypeError('img should be numpy.ndarray. Got {}'.format(type(img)))


def is_rgb(img):
    """
    Check if the input image is RGB.

    Args:
        img (numpy.ndarray): Image to be checked.

    Returns:
        Bool, True if input is RGB.
    """
    if is_numpy(img):
        img_shape = np.shape(img)
        if len(np.shape(img)) == 3 and (img_shape[0] == 3 or img_shape[2] == 3):
            return True
        return False
    raise TypeError('img should be numpy.ndarray. Got {}'.format(type(img)))


def is_normalized(img):
    """
    Check if the input image is normalized between 0 to 1.

    Args:
        img (numpy.ndarray): Image to be checked.

    Returns:
        Bool, True if input is normalized between 0 to 1.
    """
    if is_numpy(img):
        minimal = np.min(img)
        maximun = np.max(img)
        if minimal >= 0 and maximun <= 1:
            return True
        return False
    raise TypeError('img should be Numpy array. Got {}'.format(type(img)))


class ImageTransform:
    """
    The abstract base class for all image transform classes.
    """

    def __init__(self):
        pass

    def _check(self, image):
        """ Check image format. If input image is RGB and its shape
        is (C, H, W), it will be transposed to (H, W, C). If the value
        of the image is not normalized , it will be normalized between 0 to 1."""
        rgb = is_rgb(image)
        chw = False
        gray3dim = False
        normalized = is_normalized(image)
        if rgb:
            chw = is_chw(image)
            if chw:
                image = chw_to_hwc(image)
            else:
                image = image
        else:
            if len(np.shape(image)) == 3:
                gray3dim = True
                image = image[0]
            else:
                image = image
        if normalized:
            image = image*255
        return rgb, chw, normalized, gray3dim, np.uint8(image)

    def _original_format(self, image, chw, normalized, gray3dim):
        """ Return transformed image with original format. """
        if not is_numpy(image):
            image = np.array(image)
        if chw:
            image = hwc_to_chw(image)
        if normalized:
            image = image / 255
        if gray3dim:
            image = np.expand_dims(image, 0)
        return image

    def transform(self, image):
        pass


class Contrast(ImageTransform):
    """
    Contrast of an image.

    Args:
        factor (Union[float, int]): Control the contrast of an image. If 1.0,
            gives the original image. If 0, gives a gray image. Default: 1.
    """

    def __init__(self, factor=1):
        super(Contrast, self).__init__()
        self.set_params(factor)

    def set_params(self, factor=1, auto_param=False):
        """
        Set contrast parameters.

        Args:
            factor (Union[float, int]): Control the contrast of an image. If 1.0
                gives the original image. If 0 gives a gray image. Default: 1.
            auto_param (bool): True if auto generate parameters. Default: False.
        """
        if auto_param:
            self.factor = np.random.uniform(-5, 5)
        else:
            self.factor = check_param_multi_types('factor', factor, [int, float])

    def transform(self, image):
        """
        Transform the image.

        Args:
            image (numpy.ndarray): Original image to be transformed.

        Returns:
            numpy.ndarray, transformed image.
        """
        image = check_numpy_param('image', image)
        ori_dtype = image.dtype
        _, chw, normalized, gray3dim, image = self._check(image)
        image = to_pil(image)
        img_contrast = ImageEnhance.Contrast(image)
        trans_image = img_contrast.enhance(self.factor)
        trans_image = self._original_format(trans_image, chw, normalized,
                                            gray3dim)

        return trans_image.astype(ori_dtype)


class Brightness(ImageTransform):
    """
    Brightness of an image.

    Args:
        factor (Union[float, int]): Control the brightness of an image. If 1.0
            gives the original image. If 0 gives a black image. Default: 1.
    """

    def __init__(self, factor=1):
        super(Brightness, self).__init__()
        self.set_params(factor)

    def set_params(self, factor=1, auto_param=False):
        """
        Set brightness parameters.

        Args:
            factor (Union[float, int]): Control the brightness of an image. If 1
                gives the original image. If 0 gives a black image. Default: 1.
            auto_param (bool): True if auto generate parameters. Default: False.
        """
        if auto_param:
            self.factor = np.random.uniform(0, 5)
        else:
            self.factor = check_param_multi_types('factor', factor, [int, float])

    def transform(self, image):
        """
        Transform the image.

        Args:
            image (numpy.ndarray): Original image to be transformed.

        Returns:
            numpy.ndarray, transformed image.
        """
        image = check_numpy_param('image', image)
        ori_dtype = image.dtype
        _, chw, normalized, gray3dim, image = self._check(image)
        image = to_pil(image)
        img_contrast = ImageEnhance.Brightness(image)
        trans_image = img_contrast.enhance(self.factor)
        trans_image = self._original_format(trans_image, chw, normalized,
                                            gray3dim)
        return trans_image.astype(ori_dtype)


class Blur(ImageTransform):
    """
    Blurs the image using Gaussian blur filter.

    Args:
        radius(Union[float, int]): Blur radius, 0 means no blur. Default: 0.
    """

    def __init__(self, radius=0):
        super(Blur, self).__init__()
        self.set_params(radius)

    def set_params(self, radius=0, auto_param=False):
        """
        Set blur parameters.

        Args:
            radius (Union[float, int]): Blur radius, 0 means no blur. Default: 0.
            auto_param (bool): True if auto generate parameters. Default: False.
        """
        if auto_param:
            self.radius = np.random.uniform(-1.5, 1.5)
        else:
            self.radius = check_param_multi_types('radius', radius, [int, float])

    def transform(self, image):
        """
        Transform the image.

        Args:
            image (numpy.ndarray): Original image to be transformed.

        Returns:
            numpy.ndarray, transformed image.
        """
        image = check_numpy_param('image', image)
        ori_dtype = image.dtype
        _, chw, normalized, gray3dim, image = self._check(image)
        image = to_pil(image)
        trans_image = image.filter(ImageFilter.GaussianBlur(radius=self.radius))
        trans_image = self._original_format(trans_image, chw, normalized,
                                            gray3dim)
        return trans_image.astype(ori_dtype)


class Noise(ImageTransform):
    """
    Add noise of an image.

    Args:
        factor (float): factor is the ratio of pixels to add noise.
            If 0 gives the original image. Default 0.
    """

    def __init__(self, factor=0):
        super(Noise, self).__init__()
        self.set_params(factor)

    def set_params(self, factor=0, auto_param=False):
        """
        Set noise parameters.

        Args:
            factor (Union[float, int]): factor is the ratio of pixels to
                add noise. If 0 gives the original image. Default 0.
            auto_param (bool): True if auto generate parameters. Default: False.
        """
        if auto_param:
            self.factor = np.random.uniform(0, 1)
        else:
            self.factor = check_param_multi_types('factor', factor, [int, float])

    def transform(self, image):
        """
        Transform the image.

        Args:
            image (numpy.ndarray): Original image to be transformed.

        Returns:
            numpy.ndarray, transformed image.
        """
        image = check_numpy_param('image', image)
        ori_dtype = image.dtype
        _, chw, normalized, gray3dim, image = self._check(image)
        noise = np.random.uniform(low=-1, high=1, size=np.shape(image))
        trans_image = np.copy(image)
        threshold = 1 - self.factor
        trans_image[noise < -threshold] = 0
        trans_image[noise > threshold] = 1
        trans_image = self._original_format(trans_image, chw, normalized,
                                            gray3dim)
        return trans_image.astype(ori_dtype)


class Translate(ImageTransform):
    """
    Translate an image.

    Args:
        x_bias (Union[int, float]): X-direction translation, x = x + x_bias*image_length.
            Default: 0.
        y_bias (Union[int, float]): Y-direction translation,  y = y + y_bias*image_wide.
            Default: 0.
    """

    def __init__(self, x_bias=0, y_bias=0):
        super(Translate, self).__init__()
        self.set_params(x_bias, y_bias)

    def set_params(self, x_bias=0, y_bias=0, auto_param=False):
        """
        Set translate parameters.

        Args:
            x_bias (Union[float, int]): X-direction translation, and x_bias should be in range of (-1, 1). Default: 0.
            y_bias (Union[float, int]): Y-direction translation, and y_bias should be in range of (-1, 1). Default: 0.
            auto_param (bool): True if auto generate parameters. Default: False.
        """
        x_bias = check_param_in_range('x_bias', x_bias, -1, 1)
        y_bias = check_param_in_range('y_bias', y_bias, -1, 1)
        self.auto_param = auto_param
        if auto_param:
            self.x_bias = np.random.uniform(-0.3, 0.3)
            self.y_bias = np.random.uniform(-0.3, 0.3)
        else:
            self.x_bias = check_param_multi_types('x_bias', x_bias,
                                                  [int, float])
            self.y_bias = check_param_multi_types('y_bias', y_bias,
                                                  [int, float])

    def transform(self, image):
        """
        Transform the image.

        Args:
            image(numpy.ndarray): Original image to be transformed.

        Returns:
            numpy.ndarray, transformed image.
        """
        image = check_numpy_param('image', image)
        ori_dtype = image.dtype
        _, chw, normalized, gray3dim, image = self._check(image)
        img = to_pil(image)
        image_shape = np.shape(image)
        self.x_bias = image_shape[1]*self.x_bias
        self.y_bias = image_shape[0]*self.y_bias
        trans_image = img.transform(img.size, Image.AFFINE,
                                    (1, 0, self.x_bias, 0, 1, self.y_bias))
        trans_image = self._original_format(trans_image, chw, normalized,
                                            gray3dim)
        return trans_image.astype(ori_dtype)


class Scale(ImageTransform):
    """
    Scale an image in the middle.

    Args:
        factor_x (Union[float, int]): Rescale in X-direction, x=factor_x*x.
            Default: 1.
        factor_y (Union[float, int]): Rescale in Y-direction, y=factor_y*y.
            Default: 1.
    """

    def __init__(self, factor_x=1, factor_y=1):
        super(Scale, self).__init__()
        self.set_params(factor_x, factor_y)

    def set_params(self, factor_x=1, factor_y=1, auto_param=False):

        """
        Set scale parameters.

        Args:
            factor_x (Union[float, int]): Rescale in X-direction, x=factor_x*x.
                Default: 1.
            factor_y (Union[float, int]): Rescale in Y-direction, y=factor_y*y.
                Default: 1.
            auto_param (bool): True if auto generate parameters. Default: False.
        """
        if auto_param:
            self.factor_x = np.random.uniform(0.7, 3)
            self.factor_y = np.random.uniform(0.7, 3)
        else:
            self.factor_x = check_param_multi_types('factor_x', factor_x,
                                                    [int, float])
            self.factor_y = check_param_multi_types('factor_y', factor_y,
                                                    [int, float])

    def transform(self, image):
        """
        Transform the image.

        Args:
            image(numpy.ndarray): Original image to be transformed.

        Returns:
            numpy.ndarray, transformed image.
        """
        image = check_numpy_param('image', image)
        ori_dtype = image.dtype
        rgb, chw, normalized, gray3dim, image = self._check(image)
        if rgb:
            h, w, _ = np.shape(image)
        else:
            h, w = np.shape(image)
        move_x_centor = w / 2*(1 - self.factor_x)
        move_y_centor = h / 2*(1 - self.factor_y)
        img = to_pil(image)
        trans_image = img.transform(img.size, Image.AFFINE,
                                    (self.factor_x, 0, move_x_centor,
                                     0, self.factor_y, move_y_centor))
        trans_image = self._original_format(trans_image, chw, normalized,
                                            gray3dim)
        return trans_image.astype(ori_dtype)


class Shear(ImageTransform):
    """
    Shear an image, for each pixel (x, y) in the sheared image, the new value is
    taken from a position (x+factor_x*y, factor_y*x+y) in the origin image. Then
    the sheared image will be rescaled to fit original size.

    Args:
        factor_x (Union[float, int]): Shear factor of horizontal direction.
            Default: 0.
        factor_y (Union[float, int]): Shear factor of vertical direction.
            Default: 0.

    """

    def __init__(self, factor_x=0, factor_y=0):
        super(Shear, self).__init__()
        self.set_params(factor_x, factor_y)

    def set_params(self, factor_x=0, factor_y=0, auto_param=False):
        """
        Set shear parameters.

        Args:
            factor_x (Union[float, int]): Shear factor of horizontal direction.
                Default: 0.
            factor_y (Union[float, int]): Shear factor of vertical direction.
                Default: 0.
            auto_param (bool): True if auto generate parameters. Default: False.
        """
        if factor_x != 0 and factor_y != 0:
            msg = 'At least one of factor_x and factor_y is zero.'
            LOGGER.error(TAG, msg)
            raise ValueError(msg)
        if auto_param:
            if np.random.uniform(-1, 1) > 0:
                self.factor_x = np.random.uniform(-2, 2)
                self.factor_y = 0
            else:
                self.factor_x = 0
                self.factor_y = np.random.uniform(-2, 2)
        else:
            self.factor_x = check_param_multi_types('factor', factor_x,
                                                    [int, float])
            self.factor_y = check_param_multi_types('factor', factor_y,
                                                    [int, float])

    def transform(self, image):
        """
        Transform the image.

        Args:
            image(numpy.ndarray): Original image to be transformed.

        Returns:
            numpy.ndarray, transformed image.
        """
        image = check_numpy_param('image', image)
        ori_dtype = image.dtype
        rgb, chw, normalized, gray3dim, image = self._check(image)
        img = to_pil(image)
        if rgb:
            h, w, _ = np.shape(image)
        else:
            h, w = np.shape(image)
        if self.factor_x != 0:
            boarder_x = [0, -w, -self.factor_x*h, -w - self.factor_x*h]
            min_x = min(boarder_x)
            max_x = max(boarder_x)
            scale = (max_x - min_x) / w
            move_x_cen = (w - scale*w - scale*h*self.factor_x) / 2
            move_y_cen = h*(1 - scale) / 2
        else:
            boarder_y = [0, -h, -self.factor_y*w, -h - self.factor_y*w]
            min_y = min(boarder_y)
            max_y = max(boarder_y)
            scale = (max_y - min_y) / h
            move_y_cen = (h - scale*h - scale*w*self.factor_y) / 2
            move_x_cen = w*(1 - scale) / 2
        trans_image = img.transform(img.size, Image.AFFINE,
                                    (scale, scale*self.factor_x, move_x_cen,
                                     scale*self.factor_y, scale, move_y_cen))
        trans_image = self._original_format(trans_image, chw, normalized,
                                            gray3dim)
        return trans_image.astype(ori_dtype)


class Rotate(ImageTransform):
    """
    Rotate an image of degrees counter clockwise around its center.

    Args:
        angle(Union[float, int]): Degrees counter clockwise. Default: 0.
    """

    def __init__(self, angle=0):
        super(Rotate, self).__init__()
        self.set_params(angle)

    def set_params(self, angle=0, auto_param=False):
        """
        Set rotate parameters.

        Args:
            angle(Union[float, int]): Degrees counter clockwise. Default: 0.
            auto_param (bool): True if auto generate parameters. Default: False.
        """
        if auto_param:
            self.angle = np.random.uniform(0, 360)
        else:
            self.angle = check_param_multi_types('angle', angle, [int, float])

    def transform(self, image):
        """
        Transform the image.

        Args:
            image(numpy.ndarray): Original image to be transformed.

        Returns:
            numpy.ndarray, transformed image.
        """
        image = check_numpy_param('image', image)
        ori_dtype = image.dtype
        _, chw, normalized, gray3dim, image = self._check(image)
        img = to_pil(image)
        trans_image = img.rotate(self.angle, expand=False)
        trans_image = self._original_format(trans_image, chw, normalized,
                                            gray3dim)
        return trans_image.astype(ori_dtype)

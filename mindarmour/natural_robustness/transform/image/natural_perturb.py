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
Base class for image natural perturbation.
"""
import numpy as np

from mindspore.dataset.vision.py_transforms_util import is_numpy, hwc_to_chw
from mindarmour.utils.logger import LogUtil

LOGGER = LogUtil.get_instance()
TAG = 'Image Transformation'


def _chw_to_hwc(img):
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


def _is_hwc(img):
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


def _is_chw(img):
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


def _is_rgb(img):
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


def _is_normalized(img):
    """
    Check if the input image is normalized between 0 to 1.

    Args:
        img (numpy.ndarray): Image to be checked.

    Returns:
        Bool, True if input is normalized between 0 to 1.
    """
    if is_numpy(img):
        minimal = np.min(img)
        maximum = np.max(img)
        if minimal >= 0 and maximum <= 1:
            return True
        return False
    raise TypeError('img should be Numpy array. Got {}'.format(type(img)))


class _NaturalPerturb:
    """
    The abstract base class for all image natural perturbation classes.
    """

    def __init__(self):
        pass

    def _check(self, image):
        """
        Check image format. If input image is RGB and its shape
        is (C, H, W), it will be transposed to (H, W, C). If the value
        of the image is not normalized , it will be rescaled between 0 to 255.
        """
        rgb = _is_rgb(image)
        chw = False
        gray3dim = False
        normalized = _is_normalized(image)
        if rgb:
            chw = _is_chw(image)
            if chw:
                image = _chw_to_hwc(image)
        else:
            if len(np.shape(image)) == 3:
                gray3dim = True
                image = image[0]
        if normalized:
            image = image * 255
        return rgb, chw, normalized, gray3dim, np.uint8(image)

    def _original_format(self, image, chw, normalized, gray3dim):
        """ Return image with original format. """
        if not is_numpy(image):
            image = np.array(image)
        if chw:
            image = hwc_to_chw(image)
        if normalized:
            image = image / 255
        if gray3dim:
            image = np.expand_dims(image, 0)
        return image

    def __call__(self, image):
        pass

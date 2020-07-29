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
import random

from mindarmour.utils._check_param import check_numpy_param

class ImageTransform:
    """
    The abstract base class for all image transform classes.
    """

    def __init__(self):
        pass

    def random_param(self):
        pass

    def transform(self):
        pass


class Contrast(ImageTransform):
    """
    Contrast of an image.

    Args:
        image (numpy.ndarray): Original image to be transformed.
        mode (str): Mode used in PIL, here mode must be in ['L', 'RGB'],
            'L' means grey image.
    """

    def __init__(self, image, mode):
        super(Contrast, self).__init__()
        self.image = check_numpy_param('image', image)
        self.mode = mode

    def random_param(self):
        """ Random generate parameters. """
        self.factor = random.uniform(-5, 5)

    def transform(self):
        img = Image.fromarray(np.uint8(self.image*255), self.mode)
        img_contrast = ImageEnhance.Contrast(img)
        trans_image = img_contrast.enhance(self.factor)
        trans_image = np.array(trans_image)/255
        return trans_image


class Brightness(ImageTransform):
    """
    Brightness of an image.

    Args:
        image (numpy.ndarray): Original image to be transformed.
        mode (str): Mode used in PIL, here mode must be in ['L', 'RGB'],
            'L' means grey image.
    """

    def __init__(self, image, mode):
        super(Brightness, self).__init__()
        self.image = check_numpy_param('image', image)
        self.mode = mode

    def random_param(self):
        """ Random generate parameters. """
        self.factor = random.uniform(0, 5)

    def transform(self):
        img = Image.fromarray(np.uint8(self.image*255), self.mode)
        img_contrast = ImageEnhance.Brightness(img)
        trans_image = img_contrast.enhance(self.factor)
        trans_image = np.array(trans_image)/255
        return trans_image


class Blur(ImageTransform):
    """
    GaussianBlur of an image.

    Args:
        image (numpy.ndarray): Original image to be transformed.
        mode (str): Mode used in PIL, here mode must be in ['L', 'RGB'],
            'L' means grey image.
    """

    def __init__(self, image, mode):
        super(Blur, self).__init__()
        self.image = check_numpy_param('image', image)
        self.mode = mode

    def random_param(self):
        """ Random generate parameters. """
        self.radius = random.uniform(-1.5, 1.5)

    def transform(self):
        """ Transform the image. """
        img = Image.fromarray(np.uint8(self.image*255), self.mode)
        trans_image = img.filter(ImageFilter.GaussianBlur(radius=self.radius))
        trans_image = np.array(trans_image)/255
        return trans_image


class Noise(ImageTransform):
    """
    Add noise of an image.

    Args:
        image (numpy.ndarray): Original image to be transformed.
        mode (str): Mode used in PIL, here mode must be in ['L', 'RGB'],
            'L' means grey image.
    """

    def __init__(self, image, mode):
        super(Noise, self).__init__()
        self.image = check_numpy_param('image', image)
        self.mode = mode

    def random_param(self):
        """ random generate parameters """
        self.factor = random.uniform(0.7, 1)

    def transform(self):
        """ Random generate parameters. """
        noise = np.random.uniform(low=-1, high=1, size=self.image.shape)
        trans_image = np.copy(self.image)
        trans_image[noise < -self.factor] = 0
        trans_image[noise > self.factor] = 1
        trans_image = np.array(trans_image)
        return trans_image


class Translate(ImageTransform):
    """
    Translate an image.

    Args:
        image (numpy.ndarray): Original image to be transformed.
        mode (str): Mode used in PIL, here mode must be in ['L', 'RGB'],
            'L' means grey image.
    """

    def __init__(self, image, mode):
        super(Translate, self).__init__()
        self.image = check_numpy_param('image', image)
        self.mode = mode

    def random_param(self):
        """ Random generate parameters. """
        image_shape = np.shape(self.image)
        self.x_bias = random.uniform(-image_shape[0]/3, image_shape[0]/3)
        self.y_bias = random.uniform(-image_shape[1]/3, image_shape[1]/3)

    def transform(self):
        """ Transform the image. """
        img = Image.fromarray(np.uint8(self.image*255), self.mode)
        trans_image = img.transform(img.size, Image.AFFINE,
                                    (1, 0, self.x_bias, 0, 1, self.y_bias))
        trans_image = np.array(trans_image)/255
        return trans_image


class Scale(ImageTransform):
    """
    Scale an image.

    Args:
        image(numpy.ndarray): Original image to be transformed.
        mode(str): Mode used in PIL, here mode must be in ['L', 'RGB'],
            'L' means grey image.
    """

    def __init__(self, image, mode):
        super(Scale, self).__init__()
        self.image = check_numpy_param('image', image)
        self.mode = mode

    def random_param(self):
        """ Random generate parameters. """
        self.factor_x = random.uniform(0.7, 2)
        self.factor_y = random.uniform(0.7, 2)

    def transform(self):
        """ Transform the image. """
        img = Image.fromarray(np.uint8(self.image*255), self.mode)
        trans_image = img.transform(img.size, Image.AFFINE,
                                    (self.factor_x, 0, 0, 0, self.factor_y, 0))
        trans_image = np.array(trans_image)/255
        return trans_image


class Shear(ImageTransform):
    """
    Shear an image.

    Args:
        image (numpy.ndarray): Original image to be transformed.
        mode (str): Mode used in PIL, here mode must be in ['L', 'RGB'],
            'L' means grey image.
    """

    def __init__(self, image, mode):
        super(Shear, self).__init__()
        self.image = check_numpy_param('image', image)
        self.mode = mode

    def random_param(self):
        """ Random generate parameters. """
        self.factor = random.uniform(0, 1)

    def transform(self):
        """ Transform the image. """
        img = Image.fromarray(np.uint8(self.image*255), self.mode)
        if np.random.random() > 0.5:
            level = -self.factor
        else:
            level = self.factor
        if np.random.random() > 0.5:
            trans_image = img.transform(img.size, Image.AFFINE,
                                        (1, level, 0, 0, 1, 0))
        else:
            trans_image = img.transform(img.size, Image.AFFINE,
                                        (1, 0, 0, level, 1, 0))
        trans_image = np.array(trans_image, dtype=np.float)/255
        return trans_image


class Rotate(ImageTransform):
    """
    Rotate an image.

    Args:
        image (numpy.ndarray): Original image to be transformed.
        mode (str): Mode used in PIL, here mode must be in ['L', 'RGB'],
            'L' means grey image.
    """

    def __init__(self, image, mode):
        super(Rotate, self).__init__()
        self.image = check_numpy_param('image', image)
        self.mode = mode

    def random_param(self):
        """ Random generate parameters. """
        self.angle = random.uniform(0, 360)

    def transform(self):
        """ Transform the image. """
        img = Image.fromarray(np.uint8(self.image*255), self.mode)
        trans_image = img.rotate(self.angle)
        trans_image = np.array(trans_image)/255
        return trans_image

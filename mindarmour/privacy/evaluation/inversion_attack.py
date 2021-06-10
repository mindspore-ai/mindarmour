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
"""
Inversion Attack
"""
import numpy as np
from scipy.special import softmax

from mindspore.nn import Cell, MSELoss
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore import context

from mindarmour.utils.util import GradWrapWithLoss
from mindarmour.utils._check_param import check_param_type, check_param_multi_types, \
    check_int_positive, check_numpy_param, check_value_positive, check_equal_shape
from mindarmour.utils.logger import LogUtil
from mindarmour.utils.util import calculate_lp_distance, compute_ssim

LOGGER = LogUtil.get_instance()
LOGGER.set_level('INFO')
TAG = 'Image inversion attack'


class InversionLoss(Cell):
    """
     The loss function for inversion attack.

     Args:
         network (Cell): The network used to infer images' deep representations.
         weights (Union[list, tuple]): Weights of three sub-loss in InversionLoss, which can be adjusted to
             obtain better results.
     """
    def __init__(self, network, weights):
        super(InversionLoss, self).__init__()
        self._network = check_param_type('network', network, Cell)
        self._mse_loss = MSELoss()
        self._weights = check_param_multi_types('weights', weights, [list, tuple])
        self._get_shape = P.Shape()
        self._zeros = P.ZerosLike()
        self._device_target = context.get_context("device_target")

    def construct(self, input_data, target_features):
        """
        Calculate the inversion attack loss, which consists of three parts. Loss_1 is for evaluating the difference
        between the target deep representations and current representations; Loss_2 is for evaluating the continuity
        between adjacent pixels; Loss_3 is for regularization.

        Args:
            input_data (Tensor): The reconstructed image during inversion attack.
            target_features (Tensor): Deep representations of the original image.

        Returns:
            Tensor, inversion attack loss of the current iteration.
        """
        output = self._network(input_data)
        loss_1 = self._mse_loss(output, target_features) / self._mse_loss(target_features, self._zeros(target_features))

        data_shape = self._get_shape(input_data)
        if self._device_target == 'CPU':
            split_op_1 = P.Split(2, data_shape[2])
            split_op_2 = P.Split(3, data_shape[3])
            data_split_1 = split_op_1(input_data)
            data_split_2 = split_op_2(input_data)
            loss_2 = 0
            for i in range(1, data_shape[2]):
                loss_2 += self._mse_loss(data_split_1[i], data_split_1[i - 1])
            for j in range(1, data_shape[3]):
                loss_2 += self._mse_loss(data_split_2[j], data_split_2[j - 1])
        else:
            data_copy_1 = self._zeros(input_data)
            data_copy_2 = self._zeros(input_data)
            data_copy_1[:, :, :(data_shape[2] - 1), :] = input_data[:, :, 1:, :]
            data_copy_2[:, :, :, :(data_shape[2] - 1)] = input_data[:, :, :, 1:]
            loss_2 = self._mse_loss(input_data, data_copy_1) + self._mse_loss(input_data, data_copy_2)

        loss_3 = self._mse_loss(input_data, self._zeros(input_data))

        loss = loss_1*self._weights[0] + loss_2*self._weights[1] + loss_3*self._weights[2]
        return loss


class ImageInversionAttack:
    """
    An attack method used to reconstruct images by inverting their deep representations.

    References: `Aravindh Mahendran, Andrea Vedaldi. Understanding Deep Image Representations by Inverting Them.
    2014. <https://arxiv.org/pdf/1412.0035.pdf>`_

    Args:
        network (Cell): The network used to infer images' deep representations.
        input_shape (tuple): Data shape of single network input, which should be in accordance with the given
            network. The format of shape should be (channel, image_width, image_height).
        input_bound (Union[tuple, list]): The pixel range of original images, which should be like [minimum_pixel,
            maximum_pixel] or (minimum_pixel, maximum_pixel).
        loss_weights (Union[list, tuple]): Weights of three sub-loss in InversionLoss, which can be adjusted to
            obtain better results. Default: (1, 0.2, 5).

    Raises:
        TypeError: If the type of network is not Cell.
        ValueError: If any value of input_shape is not positive int.
        ValueError: If any value of loss_weights is not positive value.
    """
    def __init__(self, network, input_shape, input_bound, loss_weights=(1, 0.2, 5)):
        self._network = check_param_type('network', network, Cell)
        for sub_loss_weight in loss_weights:
            check_value_positive('sub_loss_weight', sub_loss_weight)
        self._loss = InversionLoss(self._network, loss_weights)
        self._input_shape = check_param_type('input_shape', input_shape, tuple)
        for shape_dim in input_shape:
            check_int_positive('shape_dim', shape_dim)
        self._input_bound = check_param_multi_types('input_bound', input_bound, [list, tuple])
        for value_bound in self._input_bound:
            check_param_multi_types('value_bound', value_bound, [float, int])
        if self._input_bound[0] > self._input_bound[1]:
            msg = 'input_bound[0] should not be larger than input_bound[1], but got them as {} and {}'.format(
                self._input_bound[0], self._input_bound[1])
            raise ValueError(msg)

    def generate(self, target_features, iters=100):
        """
        Reconstruct images based on target_features.

        Args:
            target_features (numpy.ndarray): Deep representations of original images. The first dimension of
                target_features should be img_num. It should be noted that the shape of target_features should be
                (1, dim2, dim3, ...) if img_num equals 1.
            iters (int): iteration times of inversion attack, which should be positive integers. Default: 100.

        Returns:
            numpy.ndarray, reconstructed images, which are expected to be similar to original images.

        Raises:
            TypeError: If the type of target_features is not numpy.ndarray.
            ValueError: If any value of iters is not positive int.Z

        Examples:
            >>> net = LeNet5()
            >>> inversion_attack = ImageInversionAttack(net, input_shape=(1, 32, 32), input_bound=(0, 1),
            >>> loss_weights=[1, 0.2, 5])
            >>> features = np.random.random((2, 10)).astype(np.float32)
            >>> images = inversion_attack.generate(features, iters=10)
            >>> print(images.shape)
            (2, 1, 32, 32)
        """
        target_features = check_numpy_param('target_features', target_features)
        iters = check_int_positive('iters', iters)

        # shape checking
        img_num = target_features.shape[0]
        test_input = np.random.random((img_num,) + self._input_shape).astype(np.float32)
        test_out = self._network(Tensor(test_input)).asnumpy()
        if test_out.shape != target_features.shape:
            msg = "The shape of target_features ({}) is not in accordance with the shape" \
                  " of network output({})".format(target_features.shape, test_out.shape)
            raise ValueError(msg)
        loss_net = self._loss
        loss_grad = GradWrapWithLoss(loss_net)

        inversion_images = []
        for i in range(img_num):
            target_feature_n = target_features[i]
            inversion_image_n = np.random.random((1,) + self._input_shape).astype(np.float32)*0.05
            for s in range(iters):
                x_grad = loss_grad(Tensor(inversion_image_n), Tensor(target_feature_n)).asnumpy()
                x_grad_sign = np.sign(x_grad)
                inversion_image_n -= x_grad_sign*0.01
                inversion_image_n = np.clip(inversion_image_n, self._input_bound[0], self._input_bound[1])
                current_loss = loss_net(Tensor(inversion_image_n), Tensor(target_feature_n))
                LOGGER.info(TAG, 'iteration step: {}, loss is {}'.format(s, current_loss))
            inversion_images.append(inversion_image_n)
        return np.concatenate(np.array(inversion_images))

    def evaluate(self, original_images, inversion_images, labels=None, new_network=None):
        """
        Evaluate the quality of inverted images by three index: the average L2 distance and SSIM value between
        original images and inversion images, and the average of inverted images' confidence on true labels of inverted
        inferred by a new trained network.

        Args:
            original_images (numpy.ndarray): Original images, whose shape should be (img_num, channels, img_width,
                img_height).
            inversion_images (numpy.ndarray): Inversion images, whose shape should be (img_num, channels, img_width,
                img_height).
            labels (numpy.ndarray): Ground truth labels of original images. Default: None.
            new_network (Cell): A network whose structure contains all parts of self._network, but loaded with different
                checkpoint file. Default: None.

        Returns:
            - float, l2 distance.

            - float, average ssim value.

            - Union[float, None], average confidence. It would be None if labels or new_network is None.

        Examples:
            >>> net = LeNet5()
            >>> inversion_attack = ImageInversionAttack(net, input_shape=(1, 32, 32), input_bound=(0, 1),
            >>> loss_weights=[1, 0.2, 5])
            >>> features = np.random.random((2, 10)).astype(np.float32)
            >>> inver_images = inversion_attack.generate(features, iters=10)
            >>> ori_images = np.random.random((2, 1, 32, 32))
            >>> result = inversion_attack.evaluate(ori_images, inver_images)
            >>> print(len(result))
        """
        check_numpy_param('original_images', original_images)
        check_numpy_param('inversion_images', inversion_images)
        if labels is not None:
            check_numpy_param('labels', labels)
            true_labels = np.squeeze(labels)
            if len(true_labels.shape) > 1:
                msg = 'Shape of true_labels should be (1, n) or (n,), but got {}'.format(true_labels.shape)
                raise ValueError(msg)
            if true_labels.size != original_images.shape[0]:
                msg = 'The size of true_labels should equal the number of images, but got {} and {}'.format(
                    true_labels.size, original_images.shape[0])
                raise ValueError(msg)
        if new_network is not None:
            check_param_type('new_network', new_network, Cell)
            LOGGER.info(TAG, 'Please make sure that the network you pass is loaded with different checkpoint files '
                             'compared with that of self._network.')

        img_1, img_2 = check_equal_shape('original_images', original_images, 'inversion_images', inversion_images)
        if (len(img_1.shape) != 4) or (img_1.shape[1] != 1 and img_1.shape[1] != 3):
            msg = 'The shape format of img_1 and img_2 should be (img_num, channels, img_width, img_height),' \
                  ' but got {} and {}'.format(img_1.shape, img_2.shape)
            raise ValueError(msg)

        total_l2_distance = 0
        total_ssim = 0
        img_1 = img_1.transpose(0, 2, 3, 1)
        img_2 = img_2.transpose(0, 2, 3, 1)
        for i in range(img_1.shape[0]):
            _, l2_dis, _ = calculate_lp_distance(img_1[i], img_2[i])
            total_l2_distance += l2_dis
            total_ssim += compute_ssim(img_1[i], img_2[i])
        avg_l2_dis = total_l2_distance / img_1.shape[0]
        avg_ssim = total_ssim / img_1.shape[0]
        avg_confi = None
        if (new_network is not None) and (labels is not None):
            pred_logits = new_network(Tensor(inversion_images.astype(np.float32))).asnumpy()
            logits_softmax = softmax(pred_logits, axis=1)
            avg_confi = np.mean(logits_softmax[np.arange(img_1.shape[0]), true_labels])
        return avg_l2_dis, avg_ssim, avg_confi

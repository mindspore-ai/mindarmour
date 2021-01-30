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
""" Util for MindArmour. """
import numpy as np
from scipy.ndimage.filters import convolve

from mindspore import Tensor
from mindspore.nn import Cell
from mindspore.ops.composite import GradOperation

from mindarmour.utils._check_param import check_numpy_param, check_param_multi_types, check_equal_shape

from .logger import LogUtil

LOGGER = LogUtil.get_instance()
TAG = 'util'


def jacobian_matrix(grad_wrap_net, inputs, num_classes):
    """
    Calculate the Jacobian matrix for inputs.

    Args:
        grad_wrap_net (Cell): A network wrapped by GradWrap.
        inputs (numpy.ndarray): Input samples.
        num_classes (int): Number of labels of model output.

    Returns:
        numpy.ndarray, the Jacobian matrix of inputs. (labels, batch_size, ...)

    Raises:
        ValueError: If grad_wrap_net is not a instance of class `GradWrap`.
    """
    if not isinstance(grad_wrap_net, GradWrap):
        msg = 'grad_wrap_net be and instance of class `GradWrap`.'
        LOGGER.error(TAG, msg)
        raise ValueError(msg)
    grad_wrap_net.set_train()
    grads_matrix = []
    for idx in range(num_classes):
        sens = np.zeros((inputs.shape[0], num_classes)).astype(np.float32)
        sens[:, idx] = 1.0
        grads = grad_wrap_net(Tensor(inputs), Tensor(sens))
        grads_matrix.append(grads.asnumpy())
    return np.asarray(grads_matrix)


def jacobian_matrix_for_detection(grad_wrap_net, inputs, num_boxes, num_classes):
    """
    Calculate the Jacobian matrix for inputs, specifically for object detection model.

    Args:
        grad_wrap_net (Cell): A network wrapped by GradWrap.
        inputs (numpy.ndarray): Input samples.
        num_boxes (int): Number of boxes inferred by each image.
        num_classes (int): Number of labels of model output.

    Returns:
        numpy.ndarray, the Jacobian matrix of inputs. (labels, batch_size, ...)

    Raises:
        ValueError: If grad_wrap_net is not a instance of class `GradWrap`.
    """
    if not isinstance(grad_wrap_net, GradWrap):
        msg = 'grad_wrap_net be and instance of class `GradWrap`.'
        LOGGER.error(TAG, msg)
        raise ValueError(msg)
    grad_wrap_net.set_train()
    grads_matrix = []
    inputs_tensor = tuple()
    if isinstance(inputs, tuple):
        for item in inputs:
            inputs_tensor += (Tensor(item),)
    else:
        inputs_tensor += (Tensor(inputs),)
    for idx in range(num_classes):
        batch_size = inputs[0].shape[0] if isinstance(inputs, tuple) else inputs.shape[0]
        sens = np.zeros((batch_size, num_boxes, num_classes)).astype(np.float32)
        sens[:, :, idx] = 1.0

        grads = grad_wrap_net(*(inputs_tensor), Tensor(sens))
        grads_matrix.append(grads.asnumpy())
    return np.asarray(grads_matrix)


class WithLossCell(Cell):
    """
    Wrap the network with loss function.

    Args:
        network (Cell): The target network to wrap.
        loss_fn (Function): The loss function is used for computing loss.

    Examples:
        >>> data = Tensor(np.ones([1, 1, 32, 32]).astype(np.float32)*0.01)
        >>> label = Tensor(np.ones([1, 10]).astype(np.float32))
        >>> net = NET()
        >>> loss_fn = nn.SoftmaxCrossEntropyWithLogits()
        >>> loss_net = WithLossCell(net, loss_fn)
        >>> loss_out = loss_net(data, label)
    """
    def __init__(self, network, loss_fn):
        super(WithLossCell, self).__init__()
        self._network = network
        self._loss_fn = loss_fn

    def construct(self, data, label):
        """
        Compute loss based on the wrapped loss cell.

        Args:
            data (Tensor): Tensor data to train.
            label (Tensor): Tensor label data.

        Returns:
            Tensor, compute result.
        """
        out = self._network(data)
        return self._loss_fn(out, label)


class GradWrapWithLoss(Cell):
    """
    Construct a network to compute the gradient of loss function in input space
    and weighted by `weight`.

    Args:
        network (Cell): The target network to wrap.

    Examples:
        >>> data = Tensor(np.ones([1, 1, 32, 32]).astype(np.float32)*0.01)
        >>> labels = Tensor(np.ones([1, 10]).astype(np.float32))
        >>> net = NET()
        >>> loss_fn = nn.SoftmaxCrossEntropyWithLogits()
        >>> loss_net = WithLossCell(net, loss_fn)
        >>> grad_all = GradWrapWithLoss(loss_net)
        >>> out_grad = grad_all(data, labels)
    """

    def __init__(self, network):
        super(GradWrapWithLoss, self).__init__()
        self._grad_all = GradOperation(get_all=True, sens_param=False)
        self._network = network

    def construct(self, inputs, labels):
        """
        Compute gradient of `inputs` with labels and weight.

        Args:
            inputs (Tensor): Inputs of network.
            labels (Tensor): Labels of inputs.

        Returns:
            Tensor, gradient matrix.
        """
        gout = self._grad_all(self._network)(inputs, labels)
        return gout[0]


class GradWrap(Cell):
    """
    Construct a network to compute the gradient of network outputs in input
    space and weighted by `weight`, expressed as a jacobian matrix.

    Args:
        network (Cell): The target network to wrap.

    Examples:
        >>> data = Tensor(np.ones([1, 1, 32, 32]).astype(np.float32)*0.01)
        >>> label = Tensor(np.ones([1, 10]).astype(np.float32))
        >>> num_classes = 10
        >>> sens = np.zeros((data.shape[0], num_classes)).astype(np.float32)
        >>> sens[:, 1] = 1.0
        >>> net = NET()
        >>> wrap_net = GradWrap(net)
        >>> wrap_net(data, Tensor(sens))
    """

    def __init__(self, network):
        super(GradWrap, self).__init__()
        self.grad = GradOperation(get_all=False, sens_param=True)
        self.network = network

    def construct(self, *data):
        """
        Compute jacobian matrix.

        Args:
            data (Tensor): Data consists of inputs and weight.

                - inputs: Inputs of network.

                - weight: Weight of each gradient, 'weight' has the same shape with labels.

        Returns:
            Tensor, Jacobian matrix.
        """
        gout = self.grad(self.network)(*data)
        return gout


def calculate_iou(box_i, box_j):
    """
    Calculate the intersection over union (iou) of two boxes.

    Args:
        box_i (numpy.ndarray): Coordinates of the first box, with the format as (x1, y1, x2, y2).
            (x1, y1) and (x2, y2) are coordinates of the lower left corner and the upper right corner,
            respectively.
        box_j (numpy.ndarray): Coordinates of the second box, with the format as (x1, y1, x2, y2).

    Returns:
        float, iou of two input boxes.
    """
    check_numpy_param('box_i', box_i)
    check_numpy_param('box_j', box_j)
    if box_i.shape[-1] != 4 or box_j.shape[-1] != 4:
        msg = 'The length of both coordinate arrays should be 4, bug got {} and {}.'.format(box_i.shape, box_j.shape)
        LOGGER.error(TAG, msg)
        raise ValueError(msg)
    i_x1, i_y1, i_x2, i_y2 = box_i
    j_x1, j_y1, j_x2, j_y2 = box_j
    s_i = (i_x2 - i_x1)*(i_y2 - i_y1)
    s_j = (j_x2 - j_x1)*(j_y2 - j_y1)
    inner_left_line = max(i_x1, j_x1)
    inner_right_line = min(i_x2, j_x2)
    inner_top_line = min(i_y2, j_y2)
    inner_bottom_line = max(i_y1, j_y1)
    if inner_left_line >= inner_right_line or inner_top_line <= inner_bottom_line:
        return 0
    inner_area = (inner_right_line - inner_left_line)*(inner_top_line - inner_bottom_line)
    return inner_area / (s_i + s_j - inner_area)


def to_tensor_tuple(inputs_ori):
    """Transfer inputs data into tensor type."""
    inputs_ori = check_param_multi_types('inputs_ori', inputs_ori, [np.ndarray, tuple])
    if isinstance(inputs_ori, tuple):
        inputs_tensor = tuple()
        for item in inputs_ori:
            inputs_tensor += (Tensor(item),)
    else:
        inputs_tensor = (Tensor(inputs_ori),)
    return inputs_tensor


def calculate_lp_distance(original_image, compared_image):
    """
    Calculate l0, l2 and linf distance for two images with the same shape.

    Args:
        original_image (np.ndarray): Original image.
        compared_image (np.ndarray): Another image for comparison.

    Returns:
        tuple, (l0, l2 and linf) distances between two images.

    Raises:
        TypeError: If type of original_image or type of compared_image is not numpy.ndarray.
        ValueError: If the shape of original_image and compared_image are not the same.
    """
    check_numpy_param('original_image', original_image)
    check_numpy_param('compared_image', compared_image)
    check_equal_shape('original_image', original_image, 'compared_image', compared_image)
    avoid_zero_div = 1e-14
    diff = (original_image - compared_image).flatten()
    data = original_image.flatten()
    l0_dist = np.linalg.norm(diff, ord=0) \
               / (np.linalg.norm(data, ord=0) + avoid_zero_div)
    l2_dist = np.linalg.norm(diff, ord=2) \
               / (np.linalg.norm(data, ord=2) + avoid_zero_div)
    linf_dist = np.linalg.norm(diff, ord=np.inf) \
                 / (np.linalg.norm(data, ord=np.inf) + avoid_zero_div)
    return l0_dist, l2_dist, linf_dist


def compute_ssim(img_1, img_2, kernel_sigma=1.5, kernel_width=11):
    """
    compute structural similarity between two images.

    Args:
        img_1 (numpy.ndarray): The first image to be compared. The shape of img_1 should be (img_width, img_height,
            channels).
        img_2 (numpy.ndarray): The second image to be compared. The shape of img_2 should be (img_width, img_height,
            channels).
        kernel_sigma (float): Gassian kernel param. Default: 1.5.
        kernel_width (int): Another Gassian kernel param. Default: 11.

    Returns:
        float, structural similarity.
    """
    img_1, img_2 = check_equal_shape('images_1', img_1, 'images_2', img_2)
    if len(img_1.shape) > 2:
        if (len(img_1.shape) != 3) or (img_1.shape[2] != 1 and img_1.shape[2] != 3):
            msg = 'The shape format of img_1 and img_2 should be (img_width, img_height, channels),' \
                  ' but got {} and {}'.format(img_1.shape, img_2.shape)
            raise ValueError(msg)

    if len(img_1.shape) > 2:
        total_ssim = 0
        for i in range(img_1.shape[2]):
            total_ssim += compute_ssim(img_1[:, :, i], img_2[:, :, i])
        return total_ssim / 3

    # Create gaussian kernel
    gaussian_kernel = np.zeros((kernel_width, kernel_width))
    for i in range(kernel_width):
        for j in range(kernel_width):
            gaussian_kernel[i, j] = (1 / (2*np.pi*(kernel_sigma**2)))*np.exp(
                - (((i - 5)**2) + ((j - 5)**2)) / (2*(kernel_sigma**2)))

    img_1 = img_1.astype(np.float32)
    img_2 = img_2.astype(np.float32)

    img_sq_1 = img_1**2
    img_sq_2 = img_2**2
    img_12 = img_1*img_2

    # Mean
    img_mu_1 = convolve(img_1, gaussian_kernel)
    img_mu_2 = convolve(img_2, gaussian_kernel)

    # Mean square
    img_mu_sq_1 = img_mu_1**2
    img_mu_sq_2 = img_mu_2**2
    img_mu_12 = img_mu_1*img_mu_2

    # Variances
    img_sigma_sq_1 = convolve(img_sq_1, gaussian_kernel)
    img_sigma_sq_2 = convolve(img_sq_2, gaussian_kernel)

    # Covariance
    img_sigma_12 = convolve(img_12, gaussian_kernel)

    # Centered squares of variances
    img_sigma_sq_1 = img_sigma_sq_1 - img_mu_sq_1
    img_sigma_sq_2 = img_sigma_sq_2 - img_mu_sq_2
    img_sigma_12 = img_sigma_12 - img_mu_12

    k_1 = 0.01
    k_2 = 0.03
    c_1 = (k_1*255)**2
    c_2 = (k_2*255)**2

    # Calculate ssim
    num_ssim = (2*img_mu_12 + c_1)*(2*img_sigma_12 + c_2)
    den_ssim = (img_mu_sq_1 + img_mu_sq_2 + c_1)*(img_sigma_sq_1
                                                  + img_sigma_sq_2 + c_2)
    res = np.average(num_ssim / den_ssim)
    return res

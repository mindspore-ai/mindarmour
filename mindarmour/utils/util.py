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
from scipy.ndimage import uniform_filter

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
        sens = np.zeros((batch_size, num_boxes, num_classes), np.float32)
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
        - float, l0 distances between two images.

        - float, l2 distances between two images.

        - float, linf distances between two images.

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


def _crop(arr, crop_width):
    """Crop arr by crop_width along each dimension."""
    arr = np.array(arr, copy=False)

    if isinstance(crop_width, int):
        crops = [[crop_width, crop_width]]*arr.ndim
    elif isinstance(crop_width[0], int):
        if len(crop_width) == 1:
            crops = [[crop_width[0], crop_width[0]]]*arr.ndim
        else:
            crops = [crop_width]*arr.ndim
    elif len(crop_width) == 1:
        crops = [crop_width[0]]*arr.ndim
    elif len(crop_width) == arr.ndim:
        crops = crop_width
    else:
        msg = 'crop_width should be a sequence of N pairs, ' \
              'a single pair, or a single integer'
        LOGGER.error(TAG, msg)
        raise ValueError(msg)

    slices = tuple(slice(a, arr.shape[i] - b) for i, (a, b) in enumerate(crops))

    cropped = arr[slices]
    return cropped


def compute_ssim(image1, image2):
    """
    compute structural similarity between two images.

    Args:
        image1 (numpy.ndarray): The first image to be compared.
        image2 (numpy.ndarray): The second image to be compared.

    Returns:
        float, structural similarity.
    """
    if not image1.shape == image2.shape:
        msg = 'Input images must have the same dimensions, but got ' \
              'image1.shape: {} and image2.shape: {}' \
            .format(image1.shape, image2.shape)
        LOGGER.error(TAG, msg)
        raise ValueError()
    if len(image1.shape) == 3:  # rgb mode
        if image1.shape[0] in [1, 3]:  # from nhw to hwn
            image1 = np.array(image1).transpose(1, 2, 0)
            image2 = np.array(image2).transpose(1, 2, 0)
        # loop over channels
        n_channels = image1.shape[-1]
        total_ssim = np.empty(n_channels)
        for ch in range(n_channels):
            ch_result = compute_ssim(image1[..., ch], image2[..., ch])
            total_ssim[..., ch] = ch_result
        return total_ssim.mean()

    k1 = 0.01
    k2 = 0.03
    win_size = 7

    if np.any((np.asarray(image1.shape) - win_size) < 0):
        msg = 'Size of each dimension must be larger win_size:7, ' \
              'but got image.shape:{}.' \
            .format(image1.shape)
        LOGGER.error(TAG, msg)
        raise ValueError(msg)

    image1 = image1.astype(np.float64)
    image2 = image2.astype(np.float64)

    ndim = image1.ndim
    tmp = win_size ** ndim
    cov_norm = tmp / (tmp - 1)

    # compute means
    ux = uniform_filter(image1, size=win_size)
    uy = uniform_filter(image2, size=win_size)

    # compute variances and covariances
    uxx = uniform_filter(image1*image1, size=win_size)
    uyy = uniform_filter(image2*image2, size=win_size)
    uxy = uniform_filter(image1*image2, size=win_size)

    vx = cov_norm*(uxx - ux*ux)
    vy = cov_norm*(uyy - uy*uy)
    vxy = cov_norm*(uxy - ux*uy)

    data_range = 2
    c1 = (k1*data_range)**2
    c2 = (k2*data_range)**2

    a1 = 2*ux*uy + c1
    a2 = 2*vxy + c2
    b1 = ux**2 + uy**2 + c1
    b2 = vx + vy + c2

    d = b1*b2
    s = (a1*a2) / d

    # padding
    pad = (win_size - 1) // 2
    mean_ssim = _crop(s, pad).mean()
    return mean_ssim

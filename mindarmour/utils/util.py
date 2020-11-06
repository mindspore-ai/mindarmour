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
from mindspore import Tensor
from mindspore.nn import Cell
from mindspore.ops.composite import GradOperation

from mindarmour.utils._check_param import check_numpy_param

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

    def construct(self, inputs, weight):
        """
        Compute jacobian matrix.

        Args:
            inputs (Tensor): Inputs of network.
            weight (Tensor): Weight of each gradient, `weight` has the same
                shape with labels.

        Returns:
            Tensor, Jacobian matrix.
        """
        gout = self.grad(self.network)(inputs, weight)
        return gout


def calculate_iou(box_i, box_j):
    """
    Calculate the intersection over union (iou) of two boxes.

    Args:
        box_i (numpy.ndarray): Coordinates of the first box, with the format as (x1, y1, x2, y2).
            (x1, y1) and (x2, y2) are coordinates of the lower left corner and the upper right corner,
            respectively.
        box_j: (numpy.ndarray): Coordinates of the second box, with the format as (x1, y1, x2, y2).

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

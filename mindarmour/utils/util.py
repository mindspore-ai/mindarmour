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

from mindarmour.utils.logger import LogUtil

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
        >>> label = Tensor(np.ones([1, 10]).astype(np.float32))
        >>> net = NET()
        >>> loss_fn = nn.SoftmaxCrossEntropyWithLogits()
        >>> loss_net = WithLossCell(net, loss_fn)
        >>> grad_all = GradWrapWithLoss(loss_net)
        >>> out_grad = grad_all(data, labels)
    """

    def __init__(self, network):
        super(GradWrapWithLoss, self).__init__()
        self._grad_all = GradOperation(name="get_all",
                                       get_all=True,
                                       sens_param=False)
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
        self.grad = GradOperation(name="grad", get_all=False,
                                  sens_param=True)
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

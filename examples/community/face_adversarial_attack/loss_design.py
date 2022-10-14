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
# ============================================================================
"""optimization Settings"""
import mindspore
from mindspore import ops, nn, Tensor
from mindspore.dataset.vision.py_transforms import ToTensor
import mindspore.dataset.vision.py_transforms as P


class MyTrainOneStepCell(nn.TrainOneStepCell):
    """
    Encapsulation class of network training.

    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.

    Args:
        network (Cell): The training network. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        sens (Number): The adjust parameter. Default: 1.0.
    """

    def __init__(self, network, optimizer, sens=1.0):
        super(MyTrainOneStepCell, self).__init__(network, optimizer, sens)
        self.grad = ops.composite.GradOperation(get_all=True, sens_param=False)

    def construct(self, *inputs):
        """Defines the computation performed."""
        loss = self.network(*inputs)
        grads = self.grad(self.network)(*inputs)
        self.optimizer(grads)
        return loss


class MyWithLossCellTargetAttack(nn.Cell):
    """The loss function defined by the target attack"""
    def __init__(self, net, loss_fn, input_tensor):
        super(MyWithLossCellTargetAttack, self).__init__(auto_prefix=False)
        self.net = net
        self._loss_fn = loss_fn
        self.std = Tensor([0.229, 0.224, 0.225])
        self.mean = Tensor([0.485, 0.456, 0.406])
        self.expand_dims = mindspore.ops.ExpandDims()
        self.normalize = P.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.tensorize = ToTensor()
        self.input_tensor = input_tensor
        self.input_emb = self.net(self.expand_dims(self.input_tensor, 0))

    @property
    def backbone_network(self):
        return self.net

    def construct(self, mask_tensor):
        ref = mask_tensor
        adversarial_tensor = mindspore.numpy.where(
            (ref == 0),
            self.input_tensor,
            (mask_tensor - self.mean[:, None, None]) / self.std[:, None, None])
        adversarial_emb = self.net(self.expand_dims(adversarial_tensor, 0))
        loss = self._loss_fn(adversarial_emb)
        return loss


class MyWithLossCellNonTargetAttack(nn.Cell):
    """The loss function defined by the non target attack"""
    def __init__(self, net, loss_fn, input_tensor):
        super(MyWithLossCellNonTargetAttack, self).__init__(auto_prefix=False)
        self.net = net
        self._loss_fn = loss_fn
        self.std = Tensor([0.229, 0.224, 0.225])
        self.mean = Tensor([0.485, 0.456, 0.406])
        self.expand_dims = mindspore.ops.ExpandDims()
        self.normalize = P.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.tensorize = ToTensor()
        self.input_tensor = input_tensor
        self.input_emb = self.net(self.expand_dims(self.input_tensor, 0))

    @property
    def backbone_network(self):
        return self.net

    def construct(self, mask_tensor):
        ref = mask_tensor
        adversarial_tensor = mindspore.numpy.where(
            (ref == 0),
            self.input_tensor,
            (mask_tensor - self.mean[:, None, None]) / self.std[:, None, None])
        adversarial_emb = self.net(self.expand_dims(adversarial_tensor, 0))
        loss = self._loss_fn(adversarial_emb, self.input_emb)
        return loss


class FaceLossTargetAttack(nn.Cell):
    """The loss function of the target attack"""

    def __init__(self, target_emb):
        super(FaceLossTargetAttack, self).__init__()
        self.uniformreal = ops.UniformReal(seed=2)
        self.sum = ops.ReduceSum(keep_dims=False)
        self.norm = nn.Norm(keep_dims=True)
        self.zeroslike = ops.ZerosLike()
        self.concat_op1 = ops.Concat(1)
        self.concat_op2 = ops.Concat(2)
        self.pow = ops.Pow()
        self.reduce_sum = ops.operations.ReduceSum()
        self.target_emb = target_emb
        self.abs = ops.Abs()
        self.reduce_mean = ops.ReduceMean()

    def construct(self, adversarial_emb):
        prod_sum = self.reduce_sum(adversarial_emb * self.target_emb, (1,))
        square1 = self.reduce_sum(ops.functional.square(adversarial_emb), (1,))
        square2 = self.reduce_sum(ops.functional.square(self.target_emb), (1,))
        denom = ops.functional.sqrt(square1) * ops.functional.sqrt(square2)
        loss = -(prod_sum / denom)
        return loss


class FaceLossNoTargetAttack(nn.Cell):
    """The loss function of the non-target attack"""

    def __init__(self):
        """Initialization"""
        super(FaceLossNoTargetAttack, self).__init__()
        self.uniformreal = ops.UniformReal(seed=2)
        self.sum = ops.ReduceSum(keep_dims=False)
        self.norm = nn.Norm(keep_dims=True)
        self.zeroslike = ops.ZerosLike()
        self.concat_op1 = ops.Concat(1)
        self.concat_op2 = ops.Concat(2)
        self.pow = ops.Pow()
        self.reduce_sum = ops.operations.ReduceSum()
        self.abs = ops.Abs()
        self.reduce_mean = ops.ReduceMean()

    def construct(self, adversarial_emb, input_emb):
        prod_sum = self.reduce_sum(adversarial_emb * input_emb, (1,))
        square1 = self.reduce_sum(ops.functional.square(adversarial_emb), (1,))
        square2 = self.reduce_sum(ops.functional.square(input_emb), (1,))
        denom = ops.functional.sqrt(square1) * ops.functional.sqrt(square2)
        loss = prod_sum / denom
        return loss

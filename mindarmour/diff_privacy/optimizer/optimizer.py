# Copyright 2020 Huawei Technologies Co., Ltd
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
Differential privacy optimizer.
"""
from mindspore import nn
from mindspore import Tensor
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.common import dtype as mstype

from mindarmour.diff_privacy.mechanisms.mechanisms import MechanismsFactory
from mindarmour.utils._check_param import check_int_positive

_grad_scale = C.MultitypeFuncGraph("grad_scale")
_reciprocal = P.Reciprocal()


@_grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    """ grad scaling """
    return grad * _reciprocal(scale)


class _TupleAdd(nn.Cell):
    def __init__(self):
        super(_TupleAdd, self).__init__()
        self.add = P.TensorAdd()
        self.hyper_map = C.HyperMap()

    def construct(self, input1, input2):
        """Add two tuple of data."""
        out = self.hyper_map(self.add, input1, input2)
        return out


class DPOptimizerClassFactory:
    """
    Factory class of Optimizer.

    Args:
        micro_batches (int): The number of small batches split from an original batch. Default: 2.

    Returns:
        Optimizer, Optimizer class

    Examples:
        >>> GaussianSGD = DPOptimizerClassFactory(micro_batches=2)
        >>> GaussianSGD.set_mechanisms('Gaussian', norm_bound=1.0, initial_noise_multiplier=1.5)
        >>> net_opt = GaussianSGD.create('Momentum')(params=network.trainable_params(),
        >>>                                          learning_rate=cfg.lr,
        >>>                                          momentum=cfg.momentum)
    """

    def __init__(self, micro_batches=2):
        self._mech_factory = MechanismsFactory()
        self.mech = None
        self._micro_batches = check_int_positive('micro_batches', micro_batches)

    def set_mechanisms(self, policy, *args, **kwargs):
        """
        Get noise mechanism object.

        Args:
            policy (str): Choose mechanism type.
        """
        self.mech = self._mech_factory.create(policy, *args, **kwargs)

    def create(self, policy, *args, **kwargs):
        """
        Create DP optimizer.

        Args:
            policy (str): Choose original optimizer type.

        Returns:
            Optimizer, A optimizer with DP.
        """
        if policy == 'SGD':
            cls = self._get_dp_optimizer_class(nn.SGD, self.mech, self._micro_batches, *args, **kwargs)
            return cls
        if policy == 'Momentum':
            cls = self._get_dp_optimizer_class(nn.Momentum, self.mech, self._micro_batches, *args, **kwargs)
            return cls
        if policy == 'Adam':
            cls = self._get_dp_optimizer_class(nn.Adam, self.mech, self._micro_batches, *args, **kwargs)
            return cls
        raise NameError("The {} is not implement, please choose ['SGD', 'Momentum', 'Adam']".format(policy))

    def _get_dp_optimizer_class(self, cls, mech, micro_batches):
        """
        Wrap original mindspore optimizer with `self._mech`.
        """

        class DPOptimizer(cls):
            """
            Initialize the DPOptimizerClass.

            Returns:
                Optimizer, Optimizer class.
            """

            def __init__(self, *args, **kwargs):
                super(DPOptimizer, self).__init__(*args, **kwargs)
                self._mech = mech
                self._tuple_add = _TupleAdd()
                self._hyper_map = C.HyperMap()
                self._micro_float = Tensor(micro_batches, mstype.float32)

            def construct(self, gradients):
                """
                construct a compute flow.
                """
                grad_noise = self._hyper_map(self._mech, gradients)
                grads = self._tuple_add(gradients, grad_noise)
                grads = self._hyper_map(F.partial(_grad_scale, self._micro_float), grads)
                gradients = super(DPOptimizer, self).construct(grads)
                return gradients

        return DPOptimizer

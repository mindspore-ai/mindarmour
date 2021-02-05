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

from mindarmour.utils.logger import LogUtil
from mindarmour.utils._check_param import check_int_positive
from ..mechanisms.mechanisms import NoiseMechanismsFactory
from ..mechanisms.mechanisms import _MechanismsParamsUpdater

LOGGER = LogUtil.get_instance()
TAG = 'DP optimizer'

_grad_scale = C.MultitypeFuncGraph("grad_scale")
_reciprocal = P.Reciprocal()


@_grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    """ grad scaling """
    return grad*_reciprocal(scale)


class _TupleAdd(nn.Cell):
    def __init__(self):
        super(_TupleAdd, self).__init__()
        self.add = P.Add()
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
        Optimizer, Optimizer class.

    Examples:
        >>> GaussianSGD = DPOptimizerClassFactory(micro_batches=2)
        >>> GaussianSGD.set_mechanisms('Gaussian', norm_bound=1.0, initial_noise_multiplier=1.5)
        >>> net_opt = GaussianSGD.create('Momentum')(params=network.trainable_params(),
        >>>                                          learning_rate=0.001,
        >>>                                          momentum=0.9)
    """

    def __init__(self, micro_batches=2):
        self._mech_factory = NoiseMechanismsFactory()
        self._mech = None
        self._micro_batches = check_int_positive('micro_batches', micro_batches)

    def set_mechanisms(self, policy, *args, **kwargs):
        """
        Get noise mechanism object. Policies can be 'sgd', 'momentum'
        or 'adam'. Candidate args and kwargs can be seen in class
        NoiseMechanismsFactory of mechanisms.py.

        Args:
            policy (str): Choose mechanism type.
        """
        self._mech = self._mech_factory.create(policy, *args, **kwargs)

    def create(self, policy):
        """
        Create DP optimizer. Policies can be 'sgd', 'momentum'
        or 'adam'.

        Args:
            policy (str): Choose original optimizer type.

        Returns:
            Optimizer, an optimizer with DP.
        """
        policy_ = policy.lower()
        if policy_ == 'sgd':
            dp_opt_class = self._get_dp_optimizer_class(nn.SGD)
        elif policy_ == 'momentum':
            dp_opt_class = self._get_dp_optimizer_class(nn.Momentum)
        elif policy_ == 'adam':
            dp_opt_class = self._get_dp_optimizer_class(nn.Adam)
        else:
            msg = "The policy must be in ('SGD', 'Momentum', 'Adam'), but got {}." \
                .format(policy)
            LOGGER.error(TAG, msg)
            raise NameError(msg)
        return dp_opt_class

    def _get_dp_optimizer_class(self, opt_class):
        """
        Wrap original mindspore optimizer with `self._mech`.
        """
        if self._mech is None:
            msg = 'Noise mechanism should be given through set_mechanisms(), but got None.'
            LOGGER.error(TAG, msg)
            raise ValueError(msg)
        mech = self._mech
        micro_batches = self._micro_batches

        class DPOptimizer(opt_class):
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
                self._micro_batches = Tensor(micro_batches, mstype.float32)

                self._mech_param_updater = None
                if self._mech is not None and self._mech._decay_policy is not None:
                    self._mech_param_updater = _MechanismsParamsUpdater(decay_policy=self._mech._decay_policy,
                                                                        decay_rate=self._mech._noise_decay_rate,
                                                                        cur_noise_multiplier=
                                                                        self._mech._noise_multiplier,
                                                                        init_noise_multiplier=
                                                                        self._mech._initial_noise_multiplier)

            def construct(self, gradients):
                """
                construct a compute flow.
                """
                # generate noise
                grad_noise_tuple = ()
                for grad_item in gradients:
                    grad_noise = self._mech(grad_item)
                    grad_noise_tuple = grad_noise_tuple + (grad_noise,)
                # add noise
                gradients = self._tuple_add(gradients, grad_noise_tuple)
                # div by self._micro_batches
                gradients = self._hyper_map(F.partial(_grad_scale, self._micro_batches), gradients)
                # update mech parameters
                if self._mech_param_updater is not None:
                    multiplier = self._mech_param_updater()
                    gradients = F.depend(gradients, multiplier)
                gradients = super(DPOptimizer, self).construct(gradients)
                return gradients

        return DPOptimizer

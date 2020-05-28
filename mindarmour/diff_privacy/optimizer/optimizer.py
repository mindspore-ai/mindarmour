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
import mindspore as ms
from mindspore import nn
from mindspore import Tensor

from mindarmour.diff_privacy.mechanisms.mechanisms import MechanismsFactory
from mindarmour.utils._check_param import check_int_positive


class DPOptimizerClassFactory:
    """
    Factory class of Optimizer.

    Args:
        micro_batches (int): The number of small batches split from an origianl batch. Default: None.

    Returns:
        Optimizer, Optimizer class

    Examples:
        >>> GaussianSGD = DPOptimizerClassFactory(micro_batches=2)
        >>> GaussianSGD.set_mechanisms('Gaussian', norm_bound=1.5, initial_noise_multiplier=5.0)
        >>> net_opt = GaussianSGD.create('SGD')(params=network.trainable_params(),
        >>>                                     learning_rate=cfg.lr,
        >>>                                     momentum=cfg.momentum)
    """
    def __init__(self, micro_batches=None):
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
        if policy == 'AdamWeightDecay':
            cls = self._get_dp_optimizer_class(nn.AdamWeightDecay, self.mech, self._micro_batches, *args, **kwargs)
            return cls
        if policy == 'AdamWeightDecayDynamicLR':
            cls = self._get_dp_optimizer_class(nn.AdamWeightDecayDynamicLR,
                                               self.mech,
                                               self._micro_batches,
                                               *args, **kwargs)
            return cls
        raise NameError("The {} is not implement, please choose ['SGD', 'Momentum', 'AdamWeightDecay', "
                        "'Adam', 'AdamWeightDecayDynamicLR']".format(policy))

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

            def construct(self, gradients):
                """
                construct a compute flow.
                """
                g_len = len(gradients)
                gradient_noise = list(gradients)
                for i in range(g_len):
                    gradient_noise[i] = gradient_noise[i].asnumpy()
                    gradient_noise[i] = self._mech(gradient_noise[i].shape).asnumpy() + gradient_noise[i]
                    gradient_noise[i] = gradient_noise[i] / micro_batches
                    gradient_noise[i] = Tensor(gradient_noise[i], ms.float32)
                gradients = tuple(gradient_noise)

                gradients = super(DPOptimizer, self).construct(gradients)
                return gradients
        return DPOptimizer

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
Noise Mechanisms.
"""
import numpy as np

from mindspore import Tensor
from mindspore.nn import Cell
from mindspore.ops import operations as P
from mindspore.common.parameter import Parameter
from mindspore.common import dtype as mstype

from mindarmour.utils._check_param import check_param_type
from mindarmour.utils._check_param import check_value_positive


class MechanismsFactory:
    """ Factory class of mechanisms"""

    def __init__(self):
        pass

    @staticmethod
    def create(policy, *args, **kwargs):
        """
        Args:
            policy(str): Noise generated strategy, could be 'Gaussian' or
                'AdaGaussian'. Default: 'AdaGaussian'.
            args(Union[float, str]): Parameters used for creating noise
                mechanisms.
            kwargs(Union[float, str]): Parameters used for creating noise
                mechanisms.

        Raises:
            NameError: `policy` must be in ['Gaussian', 'AdaGaussian'].
        Returns:
            Mechanisms, class of noise generated Mechanism.
        """
        if policy == 'Gaussian':
            return GaussianRandom(*args, **kwargs)
        if policy == 'AdaGaussian':
            return AdaGaussianRandom(*args, **kwargs)
        raise NameError("The {} is not implement, please choose "
                        "['Gaussian', 'AdaGaussian']".format(policy))


class Mechanisms(Cell):
    """
    Basic class of noise generated mechanism.
    """
    def construct(self, shape):
        """
        Construct function.
        """


class GaussianRandom(Mechanisms):
    """
    Gaussian noise generated mechanism.

    Args:
        norm_bound(float): Clipping bound for the l2 norm of the gradients.
            Default: 1.0.
        initial_noise_multiplier(float): Ratio of the standard deviation of
            Gaussian noise divided by the norm_bound, which will be used to
            calculate privacy spent. Default: 1.5.

    Returns:
        Tensor, generated noise.

    Examples:
        >>> shape = (3, 2, 4)
        >>> norm_bound = 1.0
        >>> initial_noise_multiplier = 1.5
        >>> net = GaussianRandom(shape, norm_bound, initial_noise_multiplier)
        >>> res = net(shape)
        >>> print(res)
    """

    def __init__(self, norm_bound=1.0, initial_noise_multiplier=1.5):
        super(GaussianRandom, self).__init__()
        self._norm_bound = check_value_positive('norm_bound', norm_bound)
        self._initial_noise_multiplier = check_value_positive('initial_noise_multiplier',
                                                              initial_noise_multiplier,)
        stddev = self._norm_bound*self._initial_noise_multiplier
        self._stddev = stddev
        self._mean = 0

    def construct(self, shape):
        """
        Generated Gaussian noise.

        Args:
            shape(tuple): The shape of gradients.

        Returns:
            Tensor, generated noise.
        """
        shape = check_param_type('shape', shape, tuple)
        noise = np.random.normal(self._mean, self._stddev, shape)
        return Tensor(noise, mstype.float32)


class AdaGaussianRandom(Mechanisms):
    """
    Adaptive Gaussian noise generated mechanism.

    Args:
        norm_bound(float): Clipping bound for the l2 norm of the gradients.
            Default: 1.5.
        initial_noise_multiplier(float): Ratio of the standard deviation of
            Gaussian noise divided by the norm_bound, which will be used to
            calculate privacy spent. Default: 5.0.
        alpha(float): Hyperparameter for controlling the noise decay.
            Default: 6e-4.
        decay_policy(str): Noise decay strategy include 'Step' and 'Time'.
            Default: 'Time'.

    Returns:
        Tensor, generated noise.

    Examples:
        >>> shape = (3, 2, 4)
        >>> norm_bound = 1.0
        >>> initial_noise_multiplier = 0.1
        >>> alpha = 0.5
        >>> decay_policy = "Time"
        >>> net = AdaGaussianRandom(norm_bound, initial_noise_multiplier,
        >>>                         alpha, decay_policy)
        >>> res = net(shape)
        >>> print(res)
    """

    def __init__(self, norm_bound=1.5, initial_noise_multiplier=5.0,
                 alpha=6e-4, decay_policy='Time'):
        super(AdaGaussianRandom, self).__init__()
        initial_noise_multiplier = check_value_positive('initial_noise_multiplier',
                                                        initial_noise_multiplier)
        initial_noise_multiplier = Tensor(np.array(initial_noise_multiplier, np.float32))
        self._initial_noise_multiplier = Parameter(initial_noise_multiplier,
                                                   name='initial_noise_multiplier')
        self._noise_multiplier = Parameter(initial_noise_multiplier,
                                           name='noise_multiplier')
        norm_bound = check_value_positive('norm_bound', norm_bound)
        self._norm_bound = Tensor(np.array(norm_bound, np.float32))

        alpha = check_param_type('alpha', alpha, float)
        self._alpha = Tensor(np.array(alpha, np.float32))

        self._decay_policy = check_param_type('decay_policy', decay_policy, str)
        self._mean = 0.0
        self._sub = P.Sub()
        self._mul = P.Mul()
        self._add = P.TensorAdd()
        self._div = P.Div()
        self._stddev = self._update_stddev()
        self._dtype = mstype.float32

    def _update_multiplier(self):
        """ Update multiplier. """
        if self._decay_policy == 'Time':
            temp = self._div(self._initial_noise_multiplier,
                             self._noise_multiplier)
            temp = self._add(temp, self._alpha)
            temp = self._div(self._initial_noise_multiplier, temp)
            self._noise_multiplier = Parameter(temp, name='noise_multiplier')
        else:
            one = Tensor(1, self._dtype)
            temp = self._sub(one, self._alpha)
            temp = self._mul(temp, self._noise_multiplier)
            self._noise_multiplier = Parameter(temp, name='noise_multiplier')

    def _update_stddev(self):
        self._stddev = self._mul(self._noise_multiplier, self._norm_bound)
        return self._stddev

    def construct(self, shape):
        """
        Generate adaptive Gaussian noise.

        Args:
            shape(tuple): The shape of gradients.

        Returns:
            Tensor, generated noise.
        """
        shape = check_param_type('shape', shape, tuple)
        noise = np.random.normal(self._mean, self._stddev.asnumpy(),
                                 shape)
        self._update_multiplier()
        self._update_stddev()
        return Tensor(noise, mstype.float32)

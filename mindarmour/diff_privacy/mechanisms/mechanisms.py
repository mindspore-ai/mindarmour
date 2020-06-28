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
from mindspore import Tensor
from mindspore.nn import Cell
from mindspore.ops import operations as P
from mindspore.common.parameter import Parameter
from mindspore.common import dtype as mstype

from mindarmour.utils._check_param import check_param_type
from mindarmour.utils._check_param import check_value_positive
from mindarmour.utils._check_param import check_value_non_negative
from mindarmour.utils._check_param import check_param_in_range


class MechanismsFactory:
    """ Factory class of mechanisms"""

    def __init__(self):
        pass

    @staticmethod
    def create(policy, *args, **kwargs):
        """
        Args:
            policy(str): Noise generated strategy, could be 'Gaussian' or
                'AdaGaussian'. Noise would be decayed with 'AdaGaussian' mechanism
                while be constant with 'Gaussian' mechanism.
            args(Union[float, str]): Parameters used for creating noise
                mechanisms.
            kwargs(Union[float, str]): Parameters used for creating noise
                mechanisms.

        Raises:
            NameError: `policy` must be in ['Gaussian', 'AdaGaussian'].

        Returns:
            Mechanisms, class of noise generated Mechanism.

        Examples:
            >>> class Net(nn.Cell):
            >>>     def __init__(self):
            >>>         super(Net, self).__init__()
            >>>         self.conv = nn.Conv2d(3, 64, 3, has_bias=False, weight_init='normal')
            >>>         self.bn = nn.BatchNorm2d(64)
            >>>         self.relu = nn.ReLU()
            >>>         self.flatten = nn.Flatten()
            >>>         self.fc = nn.Dense(64*224*224, 12) # padding=0
            >>>
            >>>     def construct(self, x):
            >>>         x = self.conv(x)
            >>>         x = self.bn(x)
            >>>         x = self.relu(x)
            >>>         x = self.flatten(x)
            >>>         out = self.fc(x)
            >>>         return out
            >>> norm_clip = 1.0
            >>> initial_noise_multiplier = 1.5
            >>> net = Net()
            >>> loss = nn.SoftmaxCrossEntropyWithLogits(is_grad=False, sparse=True)
            >>> net_opt = Momentum(params=net.trainable_params(), learning_rate=0.01, momentum=0.9)
            >>> mech = MechanismsFactory().create('Gaussian',
            >>>                                   norm_bound=norm_clip,
            >>>                                   initial_noise_multiplier=initial_noise_multiplier)
            >>> model = DPModel(micro_batches=2,
            >>>                 norm_clip=1.0,
            >>>                 mech=mech,
            >>>                 network=net,
            >>>                 loss_fn=loss,
            >>>                 optimizer=net_opt,
            >>>                 metrics=None)
            >>> dataset = get_dataset()
            >>> model.train(2, dataset)
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

    def construct(self, gradients):
        """
        Construct function.
        """


class GaussianRandom(Mechanisms):
    """
    Gaussian noise generated mechanism.

    Args:
        norm_bound(float): Clipping bound for the l2 norm of the gradients.
            Default: 0.5.
        initial_noise_multiplier(float): Ratio of the standard deviation of
            Gaussian noise divided by the norm_bound, which will be used to
            calculate privacy spent. Default: 1.5.
        mean(float): Average value of random noise. Default: 0.0.
        seed(int): Original random seed. Default: 0.

    Returns:
        Tensor, generated noise with shape like given gradients.

    Examples:
        >>> gradients = Tensor([0.2, 0.9], mstype.float32)
        >>> norm_bound = 0.5
        >>> initial_noise_multiplier = 1.5
        >>> net = GaussianRandom(norm_bound, initial_noise_multiplier)
        >>> res = net(gradients)
        >>> print(res)
    """

    def __init__(self, norm_bound=0.5, initial_noise_multiplier=1.5, mean=0.0, seed=0):
        super(GaussianRandom, self).__init__()
        self._norm_bound = check_value_positive('norm_bound', norm_bound)
        self._norm_bound = Tensor(norm_bound, mstype.float32)
        self._initial_noise_multiplier = check_value_positive('initial_noise_multiplier',
                                                              initial_noise_multiplier)
        self._initial_noise_multiplier = Tensor(initial_noise_multiplier, mstype.float32)
        mean = check_param_type('mean', mean, float)
        mean = check_value_non_negative('mean', mean)
        self._mean = Tensor(mean, mstype.float32)
        self._normal = P.Normal(seed=seed)

    def construct(self, gradients):
        """
        Generated Gaussian noise.

        Args:
            gradients(Tensor): The gradients.

        Returns:
            Tensor, generated noise with shape like given gradients.
        """
        shape = P.Shape()(gradients)
        stddev = P.Mul()(self._norm_bound, self._initial_noise_multiplier)
        noise = self._normal(shape, self._mean, stddev)
        return noise


class AdaGaussianRandom(Mechanisms):
    """
    Adaptive Gaussian noise generated mechanism. Noise would be decayed with training. Decay mode could be 'Time'
    mode or 'Step' mode.

    Args:
        norm_bound(float): Clipping bound for the l2 norm of the gradients.
            Default: 1.0.
        initial_noise_multiplier(float): Ratio of the standard deviation of
            Gaussian noise divided by the norm_bound, which will be used to
            calculate privacy spent. Default: 1.5.
        mean(float): Average value of random noise. Default: 0.0
        noise_decay_rate(float): Hyper parameter for controlling the noise decay.
            Default: 6e-4.
        decay_policy(str): Noise decay strategy include 'Step' and 'Time'.
            Default: 'Time'.
        seed(int): Original random seed. Default: 0.

    Returns:
        Tensor, generated noise with shape like given gradients.

    Examples:
        >>> gradients = Tensor([0.2, 0.9], mstype.float32)
        >>> norm_bound = 1.0
        >>> initial_noise_multiplier = 1.5
        >>> mean = 0.0
        >>> noise_decay_rate = 6e-4
        >>> decay_policy = "Time"
        >>> net = AdaGaussianRandom(norm_bound, initial_noise_multiplier, mean
        >>>                         noise_decay_rate, decay_policy)
        >>> res = net(gradients)
        >>> print(res)
    """

    def __init__(self, norm_bound=1.0, initial_noise_multiplier=1.5, mean=0.0,
                 noise_decay_rate=6e-4, decay_policy='Time', seed=0):
        super(AdaGaussianRandom, self).__init__()
        norm_bound = check_value_positive('norm_bound', norm_bound)
        initial_noise_multiplier = check_value_positive('initial_noise_multiplier',
                                                        initial_noise_multiplier)
        self._norm_bound = Tensor(norm_bound, mstype.float32)

        initial_noise_multiplier = Tensor(initial_noise_multiplier, mstype.float32)
        self._initial_noise_multiplier = Parameter(initial_noise_multiplier,
                                                   name='initial_noise_multiplier')
        self._stddev = P.Mul()(self._norm_bound, self._initial_noise_multiplier)
        self._noise_multiplier = Parameter(initial_noise_multiplier,
                                           name='noise_multiplier')
        mean = check_param_type('mean', mean, float)
        mean = check_value_non_negative('mean', mean)
        self._mean = Tensor(mean, mstype.float32)
        noise_decay_rate = check_param_type('noise_decay_rate', noise_decay_rate, float)
        check_param_in_range('noise_decay_rate', noise_decay_rate, 0.0, 1.0)
        self._noise_decay_rate = Tensor(noise_decay_rate, mstype.float32)
        if decay_policy not in ['Time', 'Step']:
            raise NameError("The decay_policy must be in ['Time', 'Step'], but "
                            "get {}".format(decay_policy))
        self._decay_policy = decay_policy
        self._sub = P.Sub()
        self._mul = P.Mul()
        self._add = P.TensorAdd()
        self._div = P.Div()
        self._dtype = mstype.float32
        self._normal = P.Normal(seed=seed)
        self._assign = P.Assign()

    def _update_multiplier(self):
        """ Update multiplier. """
        if self._decay_policy == 'Time':
            temp = self._div(self._initial_noise_multiplier,
                             self._noise_multiplier)
            temp = self._add(temp, self._noise_decay_rate)
            self._noise_multiplier = self._assign(self._noise_multiplier,
                                                  self._div(self._initial_noise_multiplier, temp))
        else:
            one = Tensor(1, self._dtype)
            temp = self._sub(one, self._noise_decay_rate)
            self._noise_multiplier = self._assign(self._noise_multiplier, self._mul(temp, self._noise_multiplier))
        return self._noise_multiplier

    def _update_stddev(self):
        self._stddev = self._assign(self._stddev, self._mul(self._noise_multiplier, self._norm_bound))
        return self._stddev

    def construct(self, gradients):
        """
        Generate adaptive Gaussian noise.

        Args:
            gradients(Tensor): The gradients.

        Returns:
            Tensor, generated noise with shape like given gradients.
        """
        shape = P.Shape()(gradients)
        noise = self._normal(shape, self._mean, self._stddev)
        # pylint: disable=unused-variable
        mt = self._update_multiplier()
        # pylint: disable=unused-variable
        std = self._update_stddev()
        return noise

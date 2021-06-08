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
from abc import abstractmethod

from mindspore import Tensor
from mindspore.nn import Cell
from mindspore.ops import operations as P
from mindspore.ops.composite import normal
from mindspore.common.parameter import Parameter
from mindspore.common import dtype as mstype

from mindarmour.utils._check_param import check_param_type
from mindarmour.utils._check_param import check_value_positive
from mindarmour.utils._check_param import check_param_in_range
from mindarmour.utils._check_param import check_value_non_negative
from mindarmour.utils.logger import LogUtil

LOGGER = LogUtil.get_instance()
TAG = 'NoiseMechanism'


class ClipMechanismsFactory:
    """ Factory class of clip mechanisms"""

    def __init__(self):
        pass

    @staticmethod
    def create(mech_name, decay_policy='Linear', learning_rate=0.001,
               target_unclipped_quantile=0.9, fraction_stddev=0.01, seed=0):
        """
        Args:
            mech_name(str): Clip noise generated strategy, support 'Gaussian' now.
            decay_policy(str): Decay policy of adaptive clipping, decay_policy must
                be in ['Linear', 'Geometric']. Default: Linear.
            learning_rate(float): Learning rate of update norm clip. Default: 0.001.
            target_unclipped_quantile(float): Target quantile of norm clip. Default: 0.9.
            fraction_stddev(float): The stddev of Gaussian normal which used in
                empirical_fraction, the formula is :math:`empirical fraction + N(0, fraction sstddev)`.
                Default: 0.01.
            seed(int): Original random seed, if seed=0 random normal will use secure
                random number. IF seed!=0 random normal will generate values using
                given seed. Default: 0.

        Raises:
            NameError: `mech_name` must be in ['Gaussian'].

        Returns:
            Mechanisms, class of noise generated Mechanism.

        Examples:
            >>> decay_policy = 'Linear'
            >>> beta = Tensor(0.5, mstype.float32)
            >>> norm_bound = Tensor(1.0, mstype.float32)
            >>> beta_stddev = 0.01
            >>> learning_rate = 0.001
            >>> target_unclipped_quantile = 0.9
            >>> clip_mechanism = ClipMechanismsFactory()
            >>> ada_clip = clip_mechanism.create('Gaussian',
            >>>                          decay_policy=decay_policy,
            >>>                          learning_rate=learning_rate,
            >>>                          target_unclipped_quantile=target_unclipped_quantile,
            >>>                          fraction_stddev=beta_stddev)
            >>> next_norm_bound = ada_clip(beta, norm_bound)

        """
        if mech_name == 'Gaussian':
            return AdaClippingWithGaussianRandom(decay_policy, learning_rate,
                                                 target_unclipped_quantile, fraction_stddev, seed)
        raise NameError("The {} is not implement, please choose "
                        "['Gaussian']".format(mech_name))


class NoiseMechanismsFactory:
    """ Factory class of noise mechanisms"""

    def __init__(self):
        pass

    @staticmethod
    def create(mech_name, norm_bound=1.0, initial_noise_multiplier=1.0, seed=0, noise_decay_rate=6e-6,
               decay_policy=None):
        """
        Args:
            mech_name(str): Noise generated strategy, could be 'Gaussian' or
                'AdaGaussian'. Noise would be decayed with 'AdaGaussian' mechanism
                while be constant with 'Gaussian' mechanism.
            norm_bound(float): Clipping bound for the l2 norm of the gradients. Default: 1.0.
            initial_noise_multiplier(float): Ratio of the standard deviation of
                Gaussian noise divided by the norm_bound, which will be used to
                calculate privacy spent. Default: 1.0.
            seed(int): Original random seed, if seed=0 random normal will use secure
                random number. IF seed!=0 random normal will generate values using
                given seed. Default: 0.
            noise_decay_rate(float): Hyper parameter for controlling the noise decay. Default: 6e-6.
            decay_policy(str): Mechanisms parameters update policy. If decay_policy is None, no
                parameters need update. Default: None.

        Raises:
            NameError: `mech_name` must be in ['Gaussian', 'AdaGaussian'].

        Returns:
            Mechanisms, class of noise generated Mechanism.

        Examples:
            >>> norm_bound = 1.0
            >>> initial_noise_multiplier = 1.0
            >>> network = LeNet5()
            >>> batch_size = 32
            >>> batches = 128
            >>> epochs = 1
            >>> loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
            >>> noise_mech = NoiseMechanismsFactory().create('Gaussian',
            >>>                                              norm_bound=norm_bound,
            >>>                                              initial_noise_multiplier=initial_noise_multiplier)
            >>> clip_mech = ClipMechanismsFactory().create('Gaussian',
            >>>                                            decay_policy='Linear',
            >>>                                            learning_rate=0.001,
            >>>                                            target_unclipped_quantile=0.9,
            >>>                                            fraction_stddev=0.01)
            >>> net_opt = nn.Momentum(network.trainable_params(), learning_rate=0.1,
            >>>                       momentum=0.9)
            >>> model = DPModel(micro_batches=2,
            >>>                 clip_mech=clip_mech,
            >>>                 norm_bound=norm_bound,
            >>>                 noise_mech=noise_mech,
            >>>                 network=network,
            >>>                 loss_fn=loss,
            >>>                 optimizer=net_opt,
            >>>                 metrics=None)
            >>> ms_ds = ds.GeneratorDataset(dataset_generator,
            >>>                            ['data', 'label'])
            >>> model.train(epochs, ms_ds, dataset_sink_mode=False)
        """
        if mech_name == 'Gaussian':
            return NoiseGaussianRandom(norm_bound=norm_bound,
                                       initial_noise_multiplier=initial_noise_multiplier,
                                       seed=seed,
                                       decay_policy=decay_policy)
        if mech_name == 'AdaGaussian':
            return NoiseAdaGaussianRandom(norm_bound=norm_bound,
                                          initial_noise_multiplier=initial_noise_multiplier,
                                          seed=seed,
                                          noise_decay_rate=noise_decay_rate,
                                          decay_policy=decay_policy)
        raise NameError("The {} is not implement, please choose "
                        "['Gaussian', 'AdaGaussian']".format(mech_name))


class _Mechanisms(Cell):
    """
    Basic class of noise generated mechanism.
    """

    @abstractmethod
    def construct(self, gradients):
        """
        Construct function.
        """


class NoiseGaussianRandom(_Mechanisms):
    """
    Gaussian noise generated mechanism.

    Args:
        norm_bound(float): Clipping bound for the l2 norm of the gradients.
            Default: 1.0.
        initial_noise_multiplier(float): Ratio of the standard deviation of
            Gaussian noise divided by the norm_bound, which will be used to
            calculate privacy spent. Default: 1.0.
        seed(int): Original random seed, if seed=0, random normal will use secure
            random number. If seed!=0, random normal will generate values using
            given seed. Default: 0.
        decay_policy(str): Mechanisms parameters update policy. Default: None.

    Returns:
        Tensor, generated noise with shape like given gradients.

    Examples:
        >>> gradients = Tensor([0.2, 0.9], mstype.float32)
        >>> norm_bound = 0.1
        >>> initial_noise_multiplier = 1.0
        >>> seed = 0
        >>> decay_policy = None
        >>> net = NoiseGaussianRandom(norm_bound, initial_noise_multiplier, seed, decay_policy)
        >>> res = net(gradients)
        >>> print(res)
    """

    def __init__(self, norm_bound=1.0, initial_noise_multiplier=1.0, seed=0, decay_policy=None):
        super(NoiseGaussianRandom, self).__init__()
        norm_bound = check_param_type('norm_bound', norm_bound, float)
        self._norm_bound = check_value_positive('norm_bound', norm_bound)
        self._norm_bound = Tensor(norm_bound, mstype.float32)
        initial_noise_multiplier = check_param_type('initial_noise_multiplier', initial_noise_multiplier, float)
        self._initial_noise_multiplier = check_value_positive('initial_noise_multiplier',
                                                              initial_noise_multiplier)
        self._initial_noise_multiplier = Tensor(initial_noise_multiplier, mstype.float32)
        self._mean = Tensor(0, mstype.float32)
        if decay_policy is not None:
            raise ValueError('decay_policy must be None in GaussianRandom class, but got {}.'.format(decay_policy))
        self._decay_policy = decay_policy
        seed = check_param_type('seed', seed, int)
        self._seed = check_value_non_negative('seed', seed)

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
        noise = normal(shape, self._mean, stddev, self._seed)
        return noise


class NoiseAdaGaussianRandom(NoiseGaussianRandom):
    """
    Adaptive Gaussian noise generated mechanism. Noise would be decayed with
    training. Decay mode could be 'Time' mode, 'Step' mode, 'Exp' mode.
    `self._noise_multiplier` will be update during the model.train, using
    _MechanismsParamsUpdater.

    Args:
        norm_bound(float): Clipping bound for the l2 norm of the gradients.
             Default: 1.0.
        initial_noise_multiplier(float): Ratio of the standard deviation of
            Gaussian noise divided by the norm_bound, which will be used to
            calculate privacy spent. Default: 1.0.
        seed(int): Original random seed, if seed=0 random normal will use secure
            random number. IF seed!=0 random normal will generate values using
            given seed. Default: 0.
        noise_decay_rate(float): Hyper parameter for controlling the noise decay.
            Default: 6e-6.
        decay_policy(str): Noise decay strategy include 'Step', 'Time', 'Exp'.
            Default: 'Exp'.

    Returns:
        Tensor, generated noise with shape like given gradients.

    Examples:
        >>> gradients = Tensor([0.2, 0.9], mstype.float32)
        >>> norm_bound = 1.0
        >>> initial_noise_multiplier = 1.0
        >>> seed = 0
        >>> noise_decay_rate = 6e-6
        >>> decay_policy = "Exp"
        >>> net = NoiseAdaGaussianRandom(norm_bound, initial_noise_multiplier, seed, noise_decay_rate, decay_policy)
        >>> res = net(gradients)
        >>> print(res)
    """

    def __init__(self, norm_bound=1.0, initial_noise_multiplier=1.0, seed=0, noise_decay_rate=6e-6, decay_policy='Exp'):
        super(NoiseAdaGaussianRandom, self).__init__(norm_bound=norm_bound,
                                                     initial_noise_multiplier=initial_noise_multiplier,
                                                     seed=seed)
        self._noise_multiplier = Parameter(self._initial_noise_multiplier,
                                           name='noise_multiplier')
        noise_decay_rate = check_param_type('noise_decay_rate', noise_decay_rate, float)
        check_param_in_range('noise_decay_rate', noise_decay_rate, 0.0, 1.0)
        self._noise_decay_rate = Tensor(noise_decay_rate, mstype.float32)
        if decay_policy not in ['Time', 'Step', 'Exp']:
            raise NameError("The decay_policy must be in ['Time', 'Step', 'Exp'], but "
                            "get {}".format(decay_policy))
        self._decay_policy = decay_policy

    def construct(self, gradients):
        """
        Generated Adaptive Gaussian noise.

        Args:
            gradients(Tensor): The gradients.

        Returns:
            Tensor, generated noise with shape like given gradients.
        """
        shape = P.Shape()(gradients)
        stddev = P.Mul()(self._norm_bound, self._noise_multiplier)
        noise = normal(shape, self._mean, stddev, self._seed)
        return noise


class _MechanismsParamsUpdater(Cell):
    """
    Update mechanisms parameters, the parameters will refresh in train period.

    Args:
        decay_policy(str): Pass in by the mechanisms class, mechanisms parameters
            update policy.
        decay_rate(Tensor): Pass in by the mechanisms class, hyper parameter for
            controlling the decay size.
        cur_noise_multiplier(Parameter): Pass in by the mechanisms class,
            current params value in this time.
        init_noise_multiplier(Parameter):Pass in by the mechanisms class,
            initial params value to be updated.

    Returns:
        Tuple, next params value.
    """
    def __init__(self, decay_policy, decay_rate, cur_noise_multiplier, init_noise_multiplier):
        super(_MechanismsParamsUpdater, self).__init__()
        self._decay_policy = decay_policy
        self._decay_rate = decay_rate
        self._cur_noise_multiplier = cur_noise_multiplier
        self._init_noise_multiplier = init_noise_multiplier

        self._div = P.Div()
        self._add = P.Add()
        self._assign = P.Assign()
        self._sub = P.Sub()
        self._one = Tensor(1, mstype.float32)
        self._mul = P.Mul()
        self._exp = P.Exp()

    def construct(self):
        """
        update parameters to `self._cur_params`.

        Returns:
            Tuple, next step parameters value.
        """
        if self._decay_policy == 'Time':
            temp = self._div(self._init_noise_multiplier, self._cur_noise_multiplier)
            temp = self._add(temp, self._decay_rate)
            next_noise_multiplier = self._assign(self._cur_noise_multiplier,
                                                 self._div(self._init_noise_multiplier, temp))
        elif self._decay_policy == 'Step':
            temp = self._sub(self._one, self._decay_rate)
            next_noise_multiplier = self._assign(self._cur_noise_multiplier,
                                                 self._mul(temp, self._cur_noise_multiplier))
        else:
            next_noise_multiplier = self._assign(self._cur_noise_multiplier,
                                                 self._div(self._cur_noise_multiplier, self._exp(self._decay_rate)))
        return next_noise_multiplier


class AdaClippingWithGaussianRandom(Cell):
    """
    Adaptive clipping. If `decay_policy` is 'Linear', the update formula :math:`norm bound = norm bound -
    learning rate*(beta - target unclipped quantile)`.
    If `decay_policy` is 'Geometric', the update formula is :math:`norm bound =
    norm bound*exp(-learning rate*(empirical fraction - target unclipped quantile))`.
    where beta is the empirical fraction of samples with the value at most
    `target_unclipped_quantile`.

    Args:
        decay_policy(str): Decay policy of adaptive clipping, decay_policy must
            be in ['Linear', 'Geometric']. Default: 'Linear'.
        learning_rate(float): Learning rate of update norm clip. Default: 0.001.
        target_unclipped_quantile(float): Target quantile of norm clip. Default: 0.9.
        fraction_stddev(float): The stddev of Gaussian normal which used in
            empirical_fraction, the formula is empirical_fraction + N(0, fraction_stddev).
            Default: 0.01.
        seed(int): Original random seed, if seed=0 random normal will use secure
            random number. IF seed!=0 random normal will generate values using
            given seed. Default: 0.

    Returns:
        Tensor, undated norm clip .

    Examples:
        >>> decay_policy = 'Linear'
        >>> beta = Tensor(0.5, mstype.float32)
        >>> norm_bound = Tensor(1.0, mstype.float32)
        >>> beta_stddev = 0.01
        >>> learning_rate = 0.001
        >>> target_unclipped_quantile = 0.9
        >>> ada_clip = AdaClippingWithGaussianRandom(decay_policy=decay_policy,
        >>>                                          learning_rate=learning_rate,
        >>>                                          target_unclipped_quantile=target_unclipped_quantile,
        >>>                                          fraction_stddev=beta_stddev)
        >>> next_norm_bound = ada_clip(beta, norm_bound)

    """

    def __init__(self, decay_policy='Linear', learning_rate=0.001,
                 target_unclipped_quantile=0.9, fraction_stddev=0.01, seed=0):
        super(AdaClippingWithGaussianRandom, self).__init__()
        if decay_policy not in ['Linear', 'Geometric']:
            msg = "decay policy of adaptive clip must be in ['Linear', 'Geometric'], \
                but got: {}".format(decay_policy)
            LOGGER.error(TAG, msg)
            raise ValueError(msg)
        self._decay_policy = decay_policy
        learning_rate = check_param_type('learning_rate', learning_rate, float)
        learning_rate = check_value_positive('learning_rate', learning_rate)
        self._learning_rate = Tensor(learning_rate, mstype.float32)
        fraction_stddev = check_param_type('fraction_stddev', fraction_stddev, float)
        self._fraction_stddev = Tensor(fraction_stddev, mstype.float32)
        target_unclipped_quantile = check_param_type('target_unclipped_quantile',
                                                     target_unclipped_quantile,
                                                     float)
        self._target_unclipped_quantile = Tensor(target_unclipped_quantile,
                                                 mstype.float32)

        self._zero = Tensor(0, mstype.float32)
        self._add = P.Add()
        self._sub = P.Sub()
        self._mul = P.Mul()
        self._exp = P.Exp()
        seed = check_param_type('seed', seed, int)
        self._seed = check_value_non_negative('seed', seed)

    def construct(self, empirical_fraction, norm_bound):
        """
        Update value of norm_bound.

        Args:
            empirical_fraction(Tensor): empirical fraction of samples with the
                value at most `target_unclipped_quantile`.
            norm_bound(Tensor): Clipping bound for the l2 norm of the gradients.

        Returns:
            Tensor, generated noise with shape like given gradients.
        """
        fraction_noise = normal((1,), self._zero, self._fraction_stddev, self._seed)
        empirical_fraction = self._add(empirical_fraction, fraction_noise)
        if self._decay_policy == 'Linear':
            grad_clip = self._sub(empirical_fraction,
                                  self._target_unclipped_quantile)
            next_norm_bound = self._sub(norm_bound,
                                        self._mul(self._learning_rate, grad_clip))

        else:
            grad_clip = self._sub(empirical_fraction,
                                  self._target_unclipped_quantile)
            grad_clip = self._exp(self._mul(-self._learning_rate, grad_clip))
            next_norm_bound = self._mul(norm_bound, grad_clip)
        return next_norm_bound

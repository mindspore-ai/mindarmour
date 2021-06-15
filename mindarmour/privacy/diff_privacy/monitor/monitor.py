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
""" Monitor module of differential privacy training. """
import numpy as np
from scipy import special

from mindspore.train.callback import Callback

from mindarmour.utils.logger import LogUtil
from mindarmour.utils._check_param import check_int_positive, \
    check_value_positive, check_param_in_range, check_param_type

LOGGER = LogUtil.get_instance()
TAG = 'DP monitor'


class PrivacyMonitorFactory:
    """
    Factory class of DP training's privacy monitor.
    """

    def __init__(self):
        pass

    @staticmethod
    def create(policy, *args, **kwargs):
        """
        Create a privacy monitor class.

        Args:
            policy (str): Monitor policy, 'rdp' and 'zcdp' are supported
                by now. If policy is 'rdp', the monitor will compute the
                privacy budget of DP training based on Renyi differential
                privacy theory; If policy is 'zcdp', the monitor will compute
                the privacy budget of DP training based on zero-concentrated
                differential privacy theory. It's worth noting that 'zcdp'
                is not suitable for subsampling noise mechanism.
            args (Union[int, float, numpy.ndarray, list, str]): Parameters
                used for creating a privacy monitor.
            kwargs (Union[int, float, numpy.ndarray, list, str]): Keyword
                parameters used for creating a privacy monitor.

        Returns:
            Callback, a privacy monitor.

        Examples:
            >>> rdp = PrivacyMonitorFactory.create(policy='rdp',
            >>> num_samples=60000, batch_size=32)
        """
        if policy == 'rdp':
            return RDPMonitor(*args, **kwargs)
        if policy == 'zcdp':
            return ZCDPMonitor(*args, **kwargs)
        raise ValueError("The policy must be 'rdp' or 'zcdp', but got {}".format(policy))


class RDPMonitor(Callback):
    r"""
    Compute the privacy budget of DP training based on Renyi differential
    privacy (RDP) theory. According to the reference below, if a randomized
    mechanism is said to have ε'-Renyi differential privacy of order α, it
    also satisfies conventional differential privacy (ε, δ) as below:

    .. math::
        (ε'+\frac{log(1/δ)}{α-1}, δ)

    Reference: `Rényi Differential Privacy of the Sampled Gaussian Mechanism
    <https://arxiv.org/abs/1908.10530>`_

    Args:
        num_samples (int): The total number of samples in training data sets.
        batch_size (int): The number of samples in a batch while training.
        initial_noise_multiplier (Union[float, int]): Ratio of the standard
            deviation of Gaussian noise divided by the norm_bound, which will
            be used to calculate privacy spent. Default: 1.5.
        max_eps (Union[float, int, None]): The maximum acceptable epsilon
            budget for DP training, which is used for estimating the max
            training epochs. 'None' means there is no limit to epsilon budget.
            Default: 10.0.
        target_delta (Union[float, int, None]): Target delta budget for DP
            training. If target_delta is set to be δ, then the privacy budget
            δ would be fixed during the whole training process. Default: 1e-3.
        max_delta (Union[float, int, None]): The maximum acceptable delta
            budget for DP training, which is used for estimating the max
            training epochs. Max_delta must be less than 1 and suggested
            to be less than 1e-3, otherwise overflow would be encountered.
            'None' means there is no limit to delta budget. Default: None.
        target_eps (Union[float, int, None]): Target epsilon budget for DP
            training. If target_eps is set to be ε, then the privacy budget
            ε would be fixed during the whole training process. Default: None.
        orders (Union[None, list[int, float]]): Finite orders used for
            computing rdp, which must be greater than 1. The computation result
            of privacy budget would be different for various orders. In order
            to obtain a tighter (smaller) privacy budget estimation, a list
            of orders could be tried. Default: None.
        noise_decay_mode (Union[None, str]): Decay mode of adding noise while
            training, which can be None, 'Time', 'Step' or 'Exp'. Default: 'Time'.
        noise_decay_rate (float): Decay rate of noise while training. Default: 6e-4.
        per_print_times　(int): The interval steps of computing and printing
            the privacy budget. Default: 50.
        dataset_sink_mode (bool): If True, all training data would be passed
            to device(Ascend) one-time. If False, training data would be passed
            to device after each step training. Default: False.

    Examples:
        >>> network = Net()
        >>> net_loss = nn.SoftmaxCrossEntropyWithLogits()
        >>> epochs = 2
        >>> norm_clip = 1.0
        >>> initial_noise_multiplier = 1.5
        >>> mech = NoiseMechanismsFactory().create('AdaGaussian',
        >>> norm_bound=norm_clip, initial_noise_multiplier=initial_noise_multiplier)
        >>> net_opt = nn.Momentum(network.trainable_params(), 0.01, 0.9)
        >>> model = DPModel(micro_batches=2, norm_clip=norm_clip,
        >>> mech=mech, network=network, loss_fn=loss, optimizer=net_opt, metrics=None)
        >>> rdp = PrivacyMonitorFactory.create(policy='rdp',
        >>> num_samples=60000, batch_size=256,
        >>> initial_noise_multiplier=initial_noise_multiplier)
        >>> model.train(epochs, ds, callbacks=[rdp], dataset_sink_mode=False)
    """

    def __init__(self, num_samples, batch_size, initial_noise_multiplier=1.5,
                 max_eps=10.0, target_delta=1e-3, max_delta=None,
                 target_eps=None, orders=None, noise_decay_mode='Time',
                 noise_decay_rate=6e-4, per_print_times=50, dataset_sink_mode=False):
        super(RDPMonitor, self).__init__()
        check_int_positive('num_samples', num_samples)
        check_int_positive('batch_size', batch_size)
        if batch_size >= num_samples:
            msg = 'Batch_size must be less than num_samples.'
            LOGGER.error(TAG, msg)
            raise ValueError(msg)
        check_value_positive('initial_noise_multiplier',
                             initial_noise_multiplier)
        if max_eps is not None:
            check_value_positive('max_eps', max_eps)
        if target_delta is not None:
            check_value_positive('target_delta', target_delta)
        if max_delta is not None:
            check_value_positive('max_delta', max_delta)
            if max_delta >= 1:
                msg = 'max_delta must be less than 1.'
                LOGGER.error(TAG, msg)
                raise ValueError(msg)
        if target_eps is not None:
            check_value_positive('target_eps', target_eps)
        if orders is not None:
            for item in orders:
                check_value_positive('order', item)
                if item <= 1:
                    msg = 'orders must be greater than 1'
                    LOGGER.error(TAG, msg)
                    raise ValueError(msg)
        if noise_decay_mode is not None:
            if noise_decay_mode not in ('Step', 'Time', 'Exp'):
                msg = "Noise decay mode must be in ('Step', 'Time', 'Exp')"
                LOGGER.error(TAG, msg)
                raise ValueError(msg)
            noise_decay_rate = check_param_type('noise_decay_rate', noise_decay_rate, float)
            check_param_in_range('noise_decay_rate', noise_decay_rate, 0.0, 1.0)
        check_int_positive('per_print_times', per_print_times)
        check_param_type('dataset_sink_mode', dataset_sink_mode, bool)

        self._num_samples = num_samples
        self._batch_size = batch_size
        self._initial_noise_multiplier = initial_noise_multiplier
        self._max_eps = max_eps
        self._target_delta = target_delta
        self._max_delta = max_delta
        self._target_eps = target_eps
        self._orders = orders
        self._noise_decay_mode = noise_decay_mode
        self._noise_decay_rate = noise_decay_rate
        self._rdp = 0
        self._per_print_times = per_print_times
        if self._target_eps is None and self._target_delta is None:
            msg = 'target eps and target delta cannot both be None'
            LOGGER.error(TAG, msg)
            raise ValueError(msg)

        if self._target_eps is not None and self._target_delta is not None:
            msg = 'One of target eps and target delta must be None'
            LOGGER.error(TAG, msg)
            raise ValueError(msg)

        if dataset_sink_mode:
            self._per_print_times = int(self._num_samples / self._batch_size)

    def max_epoch_suggest(self):
        """
        Estimate the maximum training epochs to satisfy the predefined
        privacy budget.

        Returns:
            int, the recommended maximum training epochs.

        Examples:
            >>> rdp = PrivacyMonitorFactory.create(policy='rdp',
            >>> num_samples=60000, batch_size=32)
            >>> suggest_epoch = rdp.max_epoch_suggest()
        """
        if self._target_delta is not None and self._max_eps is None:
            msg = 'max_eps should be consistent with target_delta, but got None.'
            LOGGER.error(TAG, msg)
            raise ValueError(msg)
        if self._target_eps is not None and self._max_delta is None:
            msg = 'max_delta should be consistent with target_eps, but got None.'
            LOGGER.error(TAG, msg)
            raise ValueError(msg)
        epoch = 1
        while epoch < 10000:
            steps = self._num_samples // self._batch_size
            eps, delta = self._compute_privacy_steps(
                list(np.arange((epoch - 1)*steps, epoch*steps + 1)))
            if self._max_eps is not None:
                if eps <= self._max_eps:
                    epoch += 1
                else:
                    break
            if self._max_delta is not None:
                if delta <= self._max_delta:
                    epoch += 1
                else:
                    break
        # reset the rdp for model training
        self._rdp = 0
        return epoch

    def step_end(self, run_context):
        """
        Compute privacy budget after each training step.

        Args:
            run_context (RunContext): Include some information of the model.
        """
        cb_params = run_context.original_args()
        cur_step = cb_params.cur_step_num
        cur_step_in_epoch = (cb_params.cur_step_num - 1) % \
                            cb_params.batch_num + 1

        if cb_params.cur_step_num % self._per_print_times == 0:
            steps = np.arange(cur_step - self._per_print_times, cur_step + 1)
            eps, delta = self._compute_privacy_steps(list(steps))
            if np.isnan(eps) or np.isinf(eps):
                msg = 'epoch: {} step: {}, invalid eps, terminating ' \
                      'training.'.format(
                          cb_params.cur_epoch_num, cur_step_in_epoch)
                LOGGER.error(TAG, msg)
                raise ValueError(msg)
            if np.isnan(delta) or np.isinf(delta):
                msg = 'epoch: {} step: {}, invalid delta, terminating ' \
                      'training.'.format(
                          cb_params.cur_epoch_num, cur_step_in_epoch)
                LOGGER.error(TAG, msg)
                raise ValueError(msg)
            print("epoch: %s step: %s, delta is %s, eps is %s" % (
                cb_params.cur_epoch_num, cur_step_in_epoch, delta, eps))

    def _compute_privacy_steps(self, steps):
        """
        Compute privacy budget corresponding to steps.

        Args:
            steps (list): Training steps.

        Returns:
            float, privacy budget.
        """
        if self._orders is None:
            self._orders = (
                [1.005, 1.01, 1.02, 1.08, 1.2, 2, 5, 10, 20, 40, 80])

        sampling_rate = self._batch_size / self._num_samples
        noise_stddev_step = self._initial_noise_multiplier

        if self._noise_decay_mode is None:
            self._rdp += self._compute_rdp(sampling_rate, noise_stddev_step)*len(
                steps)
        else:
            if self._noise_decay_mode == 'Time':
                noise_stddev_step = [self._initial_noise_multiplier / (
                    1 + self._noise_decay_rate*step) for step in steps]

            elif self._noise_decay_mode == 'Step':
                noise_stddev_step = [self._initial_noise_multiplier*(
                    1 - self._noise_decay_rate)**step for step in steps]
            elif self._noise_decay_mode == 'Exp':
                noise_stddev_step = [self._initial_noise_multiplier*np.exp(
                    -step*self._noise_decay_rate) for step in steps]
            self._rdp += sum(
                [self._compute_rdp(sampling_rate, noise) for noise in
                 noise_stddev_step])
        eps, delta = self._compute_privacy_budget(self._rdp)

        return eps, delta

    def _compute_rdp(self, sample_rate, noise_stddev):
        """
        Compute rdp according to sampling rate, added noise and Renyi
        divergence orders.

        Args:
            sample_rate (float): Sampling rate of each batch of samples.
            noise_stddev (float): Noise multiplier.

        Returns:
            float or numpy.ndarray, rdp values.
        """
        rdp = np.array(
            [_compute_rdp_with_order(sample_rate, noise_stddev, order) for order in self._orders])
        return rdp

    def _compute_privacy_budget(self, rdp):
        """
        Compute delta or eps for given rdp.

        Args:
            rdp (Union[float, numpy.ndarray]): Renyi differential privacy.

        Returns:
            float, delta budget or eps budget.
        """
        if self._target_eps is not None:
            delta = self._compute_delta(rdp)
            return self._target_eps, delta
        eps = self._compute_eps(rdp)
        return eps, self._target_delta

    def _compute_delta(self, rdp):
        """
        Compute delta for given rdp and eps.

        Args:
            rdp (Union[float, numpy.ndarray]): Renyi differential privacy.

        Returns:
            float, delta budget.
        """
        orders = np.atleast_1d(self._orders)
        rdps = np.atleast_1d(rdp)
        deltas = np.exp((rdps - self._target_eps)*(orders - 1))
        min_delta = np.min(deltas)
        return np.min([min_delta, 1.])

    def _compute_eps(self, rdp):
        """
        Compute eps for given rdp and delta.

        Args:
            rdp (Union[float, numpy.ndarray]): Renyi differential privacy.

        Returns:
            float, eps budget.
        """
        orders = np.atleast_1d(self._orders)
        rdps = np.atleast_1d(rdp)
        eps = rdps - np.log(self._target_delta) / (orders - 1)
        return np.min(eps)


class ZCDPMonitor(Callback):
    r"""
    Compute the privacy budget of DP training based on zero-concentrated
    differential privacy theory (zcdp). According to the reference below,
    if a randomized mechanism is said to have ρ-ｚCDP, it also satisfies
    conventional differential privacy (ε, δ) as below:

    .. math::
        (ρ+２\sqrt{ρ*log(1/δ)}, δ)

    It should be noted that ZCDPMonitor is not suitable for subsampling
    noise mechanisms(such as NoiseAdaGaussianRandom and NoiseGaussianRandom).
    The matching noise mechanism of ZCDP will be developed in the future.
    Reference: `Concentrated Differentially Private Gradient Descent with
    Adaptive per-Iteration Privacy Budget <https://arxiv.org/abs/1808.09501>`_

    Args:
        num_samples (int): The total number of samples in training data sets.
        batch_size (int): The number of samples in a batch while training.
        initial_noise_multiplier (Union[float, int]): Ratio of the standard
            deviation of Gaussian noise divided by the norm_bound, which will
            be used to calculate privacy spent. Default: 1.5.
        max_eps (Union[float, int]): The maximum acceptable epsilon budget for
            DP training, which is used for estimating the max training epochs.
            Default: 10.0.
        target_delta (Union[float, int]): Target delta budget for DP training.
            If target_delta is set to be δ, then the privacy budget δ would be
            fixed during the whole training process. Default: 1e-3.
        noise_decay_mode (Union[None, str]): Decay mode of adding noise while
            training, which can be None, 'Time', 'Step' or 'Exp'. Default: 'Time'.
        noise_decay_rate (float): Decay rate of noise while training. Default: 6e-4.
        per_print_times　(int): The interval steps of computing and printing
            the privacy budget. Default: 50.
        dataset_sink_mode (bool): If True, all training data would be passed
            to device(Ascend) one-time. If False, training data would be passed
            to device after each step training. Default: False.

    Examples:
        >>> network = Net()
        >>> net_loss = nn.SoftmaxCrossEntropyWithLogits()
        >>> epochs = 2
        >>> norm_clip = 1.0
        >>> initial_noise_multiplier = 1.5
        >>> mech = NoiseMechanismsFactory().create('AdaGaussian',
        >>> norm_bound=norm_clip, initial_noise_multiplier=initial_noise_multiplier)
        >>> net_opt = nn.Momentum(network.trainable_params(), 0.01, 0.9)
        >>> model = DPModel(micro_batches=2, norm_clip=norm_clip,
        >>> mech=mech, network=network, loss_fn=loss, optimizer=net_opt, metrics=None)
        >>> zcdp = PrivacyMonitorFactory.create(policy='zcdp',
        >>> num_samples=60000, batch_size=256,
        >>> initial_noise_multiplier=initial_noise_multiplier)
        >>> model.train(epochs, ds, callbacks=[zcdp], dataset_sink_mode=False)
    """

    def __init__(self, num_samples, batch_size, initial_noise_multiplier=1.5,
                 max_eps=10.0, target_delta=1e-3, noise_decay_mode='Time',
                 noise_decay_rate=6e-4, per_print_times=50, dataset_sink_mode=False):
        super(ZCDPMonitor, self).__init__()
        check_int_positive('num_samples', num_samples)
        check_int_positive('batch_size', batch_size)
        if batch_size >= num_samples:
            msg = 'Batch_size must be less than num_samples.'
            LOGGER.error(TAG, msg)
            raise ValueError(msg)
        check_value_positive('initial_noise_multiplier',
                             initial_noise_multiplier)
        if noise_decay_mode is not None:
            if noise_decay_mode not in ('Step', 'Time', 'Exp'):
                msg = "Noise decay mode must be in ('Step', 'Time', 'Exp'), but got {}.".\
                    format(noise_decay_mode)
                LOGGER.error(TAG, msg)
                raise ValueError(msg)
            noise_decay_rate = check_param_type('noise_decay_rate', noise_decay_rate, float)
            check_param_in_range('noise_decay_rate', noise_decay_rate, 0.0, 1.0)
        check_int_positive('per_print_times', per_print_times)
        check_param_type('dataset_sink_mode', dataset_sink_mode, bool)

        self._num_samples = num_samples
        self._batch_size = batch_size
        self._initial_noise_multiplier = initial_noise_multiplier
        self._max_eps = check_value_positive('max_eps', max_eps)
        self._target_delta = check_param_in_range('target_delta', target_delta, 0.0, 1.0)
        self._noise_decay_mode = noise_decay_mode
        self._noise_decay_rate = noise_decay_rate
        # initialize zcdp
        self._zcdp = 0
        self._per_print_times = per_print_times
        if dataset_sink_mode:
            self._per_print_times = int(self._num_samples / self._batch_size)

    def max_epoch_suggest(self):
        """
        Estimate the maximum training epochs to satisfy the predefined
        privacy budget.

        Returns:
            int, the recommended maximum training epochs.

        Examples:
            >>> zcdp = PrivacyMonitorFactory.create(policy='zcdp',
            >>> num_samples=60000, batch_size=32)
            >>> suggest_epoch = zcdp.max_epoch_suggest()
        """
        epoch = 1
        while epoch < 10000:
            steps = self._num_samples // self._batch_size
            eps, _ = self._compute_privacy_steps(
                list(np.arange((epoch - 1)*steps, epoch*steps + 1)))
            if eps <= self._max_eps:
                epoch += 1
            else:
                break

        # initialize the zcdp for model training
        self._zcdp = 0
        return epoch

    def step_end(self, run_context):
        """
        Compute privacy budget after each training step.

        Args:
            run_context (RunContext): Include some information of the model.
        """
        cb_params = run_context.original_args()
        cur_step = cb_params.cur_step_num
        cur_step_in_epoch = (cb_params.cur_step_num - 1) % \
                            cb_params.batch_num + 1

        if cb_params.cur_step_num % self._per_print_times == 0:
            steps = np.arange(cur_step - self._per_print_times, cur_step + 1)
            eps, delta = self._compute_privacy_steps(list(steps))
            if np.isnan(eps) or np.isinf(eps) or np.isnan(delta) or np.isinf(
                    delta):
                msg = 'epoch: {} step: {}, invalid eps, terminating ' \
                      'training.'.format(
                          cb_params.cur_epoch_num, cur_step_in_epoch)
                LOGGER.error(TAG, msg)
                raise ValueError(msg)
            print("epoch: %s step: %s, delta is %s, eps is %s" % (
                cb_params.cur_epoch_num, cur_step_in_epoch, delta, eps))

    def _compute_privacy_steps(self, steps):
        """
        Compute privacy budget corresponding to steps.

        Args:
            steps (list): Training steps.

        Returns:
            float, privacy budget.
        """
        noise_stddev_step = self._initial_noise_multiplier

        if self._noise_decay_mode is None:
            self._zcdp += self._compute_zcdp(noise_stddev_step)*len(
                steps)
        else:
            if self._noise_decay_mode == 'Time':
                noise_stddev_step = [self._initial_noise_multiplier / (
                    1 + self._noise_decay_rate*step) for step in steps]

            elif self._noise_decay_mode == 'Step':
                noise_stddev_step = [self._initial_noise_multiplier*(
                    1 - self._noise_decay_rate)**step for step in steps]
            elif self._noise_decay_mode == 'Exp':
                noise_stddev_step = [self._initial_noise_multiplier*np.exp(
                    -step*self._noise_decay_rate) for step in steps]
            self._zcdp += sum(
                [self._compute_zcdp(noise) for noise in noise_stddev_step])
        eps = self._compute_eps(self._zcdp)

        return eps, self._target_delta

    def _compute_zcdp(self, noise_stddev):
        """
        Compute zcdp according to added noise.

        Args:
            noise_stddev (float): Noise multiplier.

        Returns:
            float or numpy.ndarray, zcdp values.
        """
        zcdp = 1 / (2*noise_stddev**2)
        return zcdp

    def _compute_eps(self, zcdp):
        """
        Compute eps for given zcdp and delta.

        Args:
            zcdp (Union[float, numpy.ndarray]): zero-concentrated
            differential privacy.

        Returns:
            float, eps budget.
        """
        eps = zcdp + 2*np.sqrt(zcdp*np.log(1 / self._target_delta))
        return eps


def _compute_rdp_with_order(sample_rate, noise_stddev, order):
    """
    Compute rdp for each order.

    Args:
        sample_rate (float): Sampling probability.
        noise_stddev (float): Noise multiplier.
        order: The order used for computing rdp.

    Returns:
        float, rdp value.
    """
    if float(order).is_integer():
        log_integrate = -np.inf
        for k in range(order + 1):
            term_k = (np.log(
                special.binom(order, k)) + k*np.log(sample_rate) + (
                    order - k)*np.log(
                        1 - sample_rate)) + (k*k - k) / (2*(noise_stddev**2))
            log_integrate = _log_add(log_integrate, term_k)
        return float(log_integrate) / (order - 1)
    log_part_0, log_part_1 = -np.inf, -np.inf
    k = 0
    z0 = noise_stddev**2*np.log(1 / sample_rate - 1) + 1 / 2
    while True:
        bi_coef = special.binom(order, k)
        log_coef = np.log(abs(bi_coef))
        j = order - k

        term_k_part_0 = log_coef + k*np.log(sample_rate) + j*np.log(1 - sample_rate) + (
            k*k - k) / (2*(noise_stddev**2)) + special.log_ndtr(
                (z0 - k) / noise_stddev)

        term_k_part_1 = log_coef + j*np.log(sample_rate) + k*np.log(1 - sample_rate) + (
            j*j - j) / (2*(noise_stddev**2)) + special.log_ndtr(
                (j - z0) / noise_stddev)

        if bi_coef > 0:
            log_part_0 = _log_add(log_part_0, term_k_part_0)
            log_part_1 = _log_add(log_part_1, term_k_part_1)
        else:
            log_part_0 = _log_subtract(log_part_0, term_k_part_0)
            log_part_1 = _log_subtract(log_part_1, term_k_part_1)

        k += 1
        if np.max([term_k_part_0, term_k_part_1]) < -30:
            break

    return _log_add(log_part_0, log_part_1) / (order - 1)


def _log_add(x, y):
    """
    Add x and y in log space.
    """
    if x == -np.inf:
        return y
    if y == -np.inf:
        return x
    return np.max([x, y]) + np.log1p(np.exp(-abs(x - y)))


def _log_subtract(x, y):
    """
    Subtract y from x in log space, x must be greater than y.
    """
    if x <= y:
        msg = 'The antilog of log functions must be positive'
        LOGGER.error(TAG, msg)
        raise ValueError(msg)
    if y == -np.inf:
        return x
    return np.log1p(np.exp(y - x)) + x

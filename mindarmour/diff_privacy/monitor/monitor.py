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
import math
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
            policy (str): Monitor policy, 'rdp' is supported by now. RDP means R'enyi differential privacy,
                which computed based on R'enyi divergence.
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
        raise ValueError("Only RDP-policy is supported by now")


class RDPMonitor(Callback):
    """
    Compute the privacy budget of DP training based on Renyi differential
    privacy theory.

    Reference: `Rényi Differential Privacy of the Sampled Gaussian Mechanism
    <https://arxiv.org/abs/1908.10530>`_

    Args:
        num_samples (int): The total number of samples in training data sets.
        batch_size (int): The number of samples in a batch while training.
        initial_noise_multiplier (Union[float, int]): The initial
            multiplier of the noise added to training parameters' gradients. Default: 1.5.
        max_eps (Union[float, int, None]): The maximum acceptable epsilon
            budget for DP training. Default: 10.0.
        target_delta (Union[float, int, None]): Target delta budget for DP
            training. Default: 1e-3.
        max_delta (Union[float, int, None]): The maximum acceptable delta
            budget for DP training. Max_delta must be less than 1 and
            suggested to be less than 1e-3, otherwise overflow would be
            encountered. Default: None.
        target_eps (Union[float, int, None]): Target epsilon budget for DP
            training. Default: None.
        orders (Union[None, list[int, float]]): Finite orders used for
            computing rdp, which must be greater than 1.
        noise_decay_mode (str): Decay mode of adding noise while training,
            which can be 'no_decay', 'Time' or 'Step'. Default: 'Time'.
        noise_decay_rate (Union[float, None]): Decay rate of noise while
            training. Default: 6e-4.
        per_print_times　(int): The interval steps of computing and printing
            the privacy budget. Default: 50.

    Examples:
        >>> rdp = PrivacyMonitorFactory.create(policy='rdp',
        >>> num_samples=60000, batch_size=256)
        >>> network = Net()
        >>> net_loss = nn.SoftmaxCrossEntropyWithLogits()
        >>> net_opt = nn.Momentum(network.trainable_params(), 0.01, 0.9)
        >>> model = Model(network, net_loss, net_opt)
        >>> model.train(epochs, ds, callbacks=[rdp], dataset_sink_mode=False)
    """

    def __init__(self, num_samples, batch_size, initial_noise_multiplier=1.5,
                 max_eps=10.0, target_delta=1e-3, max_delta=None,
                 target_eps=None, orders=None, noise_decay_mode='Time',
                 noise_decay_rate=6e-4, per_print_times=50):
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
        if noise_decay_mode not in ('no_decay', 'Step', 'Time'):
            msg = "Noise decay mode must be in ('no_decay', 'Step', 'Time')"
            LOGGER.error(TAG, msg)
            raise ValueError(msg)
        if noise_decay_rate is not None:
            noise_decay_rate = check_param_type('noise_decay_rate', noise_decay_rate, float)
            check_param_in_range('noise_decay_rate', noise_decay_rate, 0.0, 1.0)
        check_int_positive('per_print_times', per_print_times)

        self._total_echo_privacy = None
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
        epoch = 1
        while epoch < 10000:
            steps = self._num_samples // self._batch_size
            eps, delta = self._compute_privacy_steps(
                list(np.arange((epoch - 1) * steps, epoch * steps + 1)))
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
            if np.isnan(eps) or np.isinf(eps) or np.isnan(delta) or np.isinf(
                    delta):
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
        if self._target_eps is None and self._target_delta is None:
            msg = 'target eps and target delta cannot both be None'
            LOGGER.error(TAG, msg)
            raise ValueError(msg)

        if self._target_eps is not None and self._target_delta is not None:
            msg = 'One of target eps and target delta must be None'
            LOGGER.error(TAG, msg)
            raise ValueError(msg)

        if self._orders is None:
            self._orders = (
                [1.005, 1.01, 1.02, 1.08, 1.2, 2, 5, 10, 20, 40, 80])

        sampling_rate = self._batch_size / self._num_samples
        noise_step = self._initial_noise_multiplier

        if self._noise_decay_mode == 'no_decay':
            self._rdp += self._compute_rdp(sampling_rate, noise_step) * len(
                steps)
        else:
            if self._noise_decay_rate is None:
                msg = 'noise_decay_rate in decay-mode cannot be None'
                LOGGER.error(TAG, msg)
                raise ValueError(msg)

            if self._noise_decay_mode == 'Time':
                noise_step = [self._initial_noise_multiplier / (
                    1 + self._noise_decay_rate * step) for step in steps]

            elif self._noise_decay_mode == 'Step':
                noise_step = [self._initial_noise_multiplier * (
                    1 - self._noise_decay_rate) ** step for step in steps]
            self._rdp += sum(
                [self._compute_rdp(sampling_rate, noise) for noise in
                 noise_step])
        eps, delta = self._compute_privacy_budget(self._rdp)

        return eps, delta

    def _compute_rdp(self, q, noise):
        """
        Compute rdp according to sampling rate, added noise and Renyi
        divergence orders.

        Args:
            q (float): Sampling rate of each batch of samples.
            noise (float): Noise multiplier.

        Returns:
            float or numpy.ndarray, rdp values.
        """
        rdp = np.array(
            [_compute_rdp_order(q, noise, order) for order in self._orders])
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
        if len(orders) != len(rdps):
            msg = 'rdp lists and orders list must have the same length.'
            LOGGER.error(TAG, msg)
            raise ValueError(msg)

        deltas = np.exp((rdps - self._target_eps) * (orders - 1))
        min_delta = min(deltas)
        return min(min_delta, 1.)

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
        if len(orders) != len(rdps):
            msg = 'rdp lists and orders list must have the same length.'
            LOGGER.error(TAG, msg)
            raise ValueError(msg)
        eps = rdps - math.log(self._target_delta) / (orders - 1)
        return min(eps)


def _compute_rdp_order(q, sigma, alpha):
    """
    Compute rdp for each order.

    Args:
        q (float): Sampling probability.
        sigma (float): Noise multiplier.
        alpha: The order used for computing rdp.

    Returns:
        float, rdp value.
    """
    if float(alpha).is_integer():
        log_integrate = -np.inf
        for k in range(alpha + 1):
            term_k = (math.log(
                special.binom(alpha, k)) + k * math.log(q) + (
                    alpha - k) * math.log(
                        1 - q)) + (k * k - k) / (2 * (sigma ** 2))
            log_integrate = _log_add(log_integrate, term_k)
        return float(log_integrate) / (alpha - 1)
    log_part_0, log_part_1 = -np.inf, -np.inf
    k = 0
    z0 = sigma ** 2 * math.log(1 / q - 1) + 1 / 2
    while True:
        bi_coef = special.binom(alpha, k)
        log_coef = math.log(abs(bi_coef))
        j = alpha - k

        term_k_part_0 = log_coef + k * math.log(q) + j * math.log(1 - q) + (
            k * k - k) / (2 * (sigma ** 2)) + special.log_ndtr(
                (z0 - k) / sigma)

        term_k_part_1 = log_coef + j * math.log(q) + k * math.log(1 - q) + (
            j * j - j) / (2 * (sigma ** 2)) + special.log_ndtr(
                (j - z0) / sigma)

        if bi_coef > 0:
            log_part_0 = _log_add(log_part_0, term_k_part_0)
            log_part_1 = _log_add(log_part_1, term_k_part_1)
        else:
            log_part_0 = _log_subtract(log_part_0, term_k_part_0)
            log_part_1 = _log_subtract(log_part_1, term_k_part_1)

        k += 1
        if max(term_k_part_0, term_k_part_1) < -30:
            break

    return _log_add(log_part_0, log_part_1) / (alpha - 1)


def _log_add(x, y):
    """
    Add x and y in log space.
    """
    if x == -np.inf:
        return y
    if y == -np.inf:
        return x
    return max(x, y) + math.log1p(math.exp(-abs(x - y)))


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
    return math.log1p(math.exp(y - x)) + x

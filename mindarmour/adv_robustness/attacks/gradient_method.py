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
Gradient-method Attack.
"""
from abc import abstractmethod

import numpy as np

from mindspore.nn import Cell

from mindarmour.utils.util import WithLossCell, GradWrapWithLoss, to_tensor_tuple
from mindarmour.utils.logger import LogUtil
from mindarmour.utils._check_param import check_model, check_inputs_labels, \
    normalize_value, check_value_positive, check_param_multi_types, \
    check_norm_level, check_param_type
from .attack import Attack

LOGGER = LogUtil.get_instance()
TAG = 'SingleGrad'


class GradientMethod(Attack):
    """
    Abstract base class for all single-step gradient-based attacks.

    Args:
        network (Cell): Target model.
        eps (float): Proportion of single-step adversarial perturbation generated
            by the attack to data range. Default: 0.07.
        alpha (float): Proportion of single-step random perturbation to data range.
            Default: None.
        bounds (tuple): Upper and lower bounds of data, indicating the data range.
            In form of (clip_min, clip_max). Default: None.
        loss_fn (Loss): Loss function for optimization. If None, the input network \
            is already equipped with loss function. Default: None.

    Examples:
        >>> inputs = np.array([[0.1, 0.2, 0.6], [0.3, 0, 0.4]])
        >>> labels = np.array([[0, 1, 0, 0, 0], [0, 0, 1, 0, 0]])
        >>> attack = FastGradientMethod(network, loss_fn=SoftmaxCrossEntropyWithLogits(sparse=False))
        >>> adv_x = attack.generate(inputs, labels)
    """

    def __init__(self, network, eps=0.07, alpha=None, bounds=None,
                 loss_fn=None):
        super(GradientMethod, self).__init__()
        self._network = check_model('network', network, Cell)
        self._eps = check_value_positive('eps', eps)
        self._dtype = None
        if bounds is not None:
            self._bounds = check_param_multi_types('bounds', bounds,
                                                   [list, tuple])
            for b in self._bounds:
                _ = check_param_multi_types('bound', b, [int, float])
        else:
            self._bounds = bounds
        if alpha is not None:
            self._alpha = check_value_positive('alpha', alpha)
        else:
            self._alpha = alpha
        if loss_fn is None:
            self._grad_all = self._network
        else:
            with_loss_cell = WithLossCell(self._network, loss_fn)
            self._grad_all = GradWrapWithLoss(with_loss_cell)
        self._grad_all.set_train()

    def generate(self, inputs, labels):
        """
        Generate adversarial examples based on input samples and original/target labels.

        Args:
            inputs (Union[numpy.ndarray, tuple]): Benign input samples used as references to create
                    adversarial examples.
            labels (Union[numpy.ndarray, tuple]): Original/target labels. \
                For each input if it has more than one label, it is wrapped in a tuple.

        Returns:
            numpy.ndarray, generated adversarial examples.
        """
        inputs_image, inputs, labels = check_inputs_labels(inputs, labels)
        self._dtype = inputs_image.dtype
        gradient = self._gradient(inputs, labels)
        # use random method or not
        if self._alpha is not None:
            random_part = self._alpha*np.sign(np.random.normal(
                size=inputs.shape)).astype(self._dtype)
            perturbation = (self._eps - self._alpha)*gradient + random_part
        else:
            perturbation = self._eps*gradient

        if self._bounds is not None:
            clip_min, clip_max = self._bounds
            perturbation = perturbation*(clip_max - clip_min)
            adv_x = inputs_image + perturbation
            adv_x = np.clip(adv_x, clip_min, clip_max)
        else:
            adv_x = inputs_image + perturbation
        return adv_x

    @abstractmethod
    def _gradient(self, inputs, labels):
        """
        Calculate gradients based on input samples and original/target labels.

        Args:
            inputs (Union[numpy.ndarray, tuple]): Benign input samples used as references to
                create adversarial examples.
            labels (Union[numpy.ndarray, tuple]): Original/target labels. \
                For each input if it has more than one label, it is wrapped in a tuple.

        Raises:
            NotImplementedError: It is an abstract method.
        """
        msg = 'The function _gradient() is an abstract method in class ' \
              '`GradientMethod`, and should be implemented in child class.'
        LOGGER.error(TAG, msg)
        raise NotImplementedError(msg)


class FastGradientMethod(GradientMethod):
    """
    This attack is a one-step attack based on gradients calculation, and
    the norm of perturbations includes L1, L2 and Linf.

    References: `I. J. Goodfellow, J. Shlens, and C. Szegedy, "Explaining
    and harnessing adversarial examples," in ICLR, 2015.
    <https://arxiv.org/abs/1412.6572>`_

    Args:
        network (Cell): Target model.
        eps (float): Proportion of single-step adversarial perturbation generated
            by the attack to data range. Default: 0.07.
        alpha (float): Proportion of single-step random perturbation to data range.
            Default: None.
        bounds (tuple): Upper and lower bounds of data, indicating the data range.
            In form of (clip_min, clip_max). Default: (0.0, 1.0).
        norm_level (Union[int, numpy.inf]): Order of the norm.
            Possible values: np.inf, 1 or 2. Default: 2.
        is_targeted (bool): If True, targeted attack. If False, untargeted
            attack. Default: False.
        loss_fn (Loss): Loss function for optimization. If None, the input network \
            is already equipped with loss function. Default: None.

    Examples:
        >>> inputs = np.array([[0.1, 0.2, 0.6], [0.3, 0, 0.4]])
        >>> labels = np.array([[0, 1, 0, 0, 0], [0, 0, 1, 0, 0]])
        >>> attack = FastGradientMethod(network, loss_fn=SoftmaxCrossEntropyWithLogits(sparse=False))
        >>> adv_x = attack.generate(inputs, labels)
    """

    def __init__(self, network, eps=0.07, alpha=None, bounds=(0.0, 1.0),
                 norm_level=2, is_targeted=False, loss_fn=None):

        super(FastGradientMethod, self).__init__(network,
                                                 eps=eps,
                                                 alpha=alpha,
                                                 bounds=bounds,
                                                 loss_fn=loss_fn)
        self._norm_level = check_norm_level(norm_level)
        self._is_targeted = check_param_type('is_targeted', is_targeted, bool)

    def _gradient(self, inputs, labels):
        """
        Calculate gradients based on input samples and original/target labels.

        Args:
            inputs (Union[numpy.ndarray, tuple]): Input sample.
            labels (Union[numpy.ndarray, tuple]): Original/target labels. \
                For each input if it has more than one label, it is wrapped in a tuple.

        Returns:
            numpy.ndarray, gradient of inputs.
        """
        inputs_tensor = to_tensor_tuple(inputs)
        labels_tensor = to_tensor_tuple(labels)
        out_grad = self._grad_all(*inputs_tensor, *labels_tensor)
        if isinstance(out_grad, tuple):
            out_grad = out_grad[0]
        gradient = out_grad.asnumpy()

        if self._is_targeted:
            gradient = -gradient
        return normalize_value(gradient, self._norm_level)


class RandomFastGradientMethod(FastGradientMethod):
    """
    Fast Gradient Method use Random perturbation.

    References: `Florian Tramer, Alexey Kurakin, Nicolas Papernot, "Ensemble
    adversarial training: Attacks and defenses" in ICLR, 2018
    <https://arxiv.org/abs/1705.07204>`_

    Args:
        network (Cell): Target model.
        eps (float): Proportion of single-step adversarial perturbation generated
            by the attack to data range. Default: 0.07.
        alpha (float): Proportion of single-step random perturbation to data range.
            Default: 0.035.
        bounds (tuple): Upper and lower bounds of data, indicating the data range.
            In form of (clip_min, clip_max). Default: (0.0, 1.0).
        norm_level (Union[int, numpy.inf]): Order of the norm.
            Possible values: np.inf, 1 or 2. Default: 2.
        is_targeted (bool): If True, targeted attack. If False, untargeted
            attack. Default: False.
        loss_fn (Loss): Loss function for optimization. If None, the input network \
            is already equipped with loss function. Default: None.

    Raises:
        ValueError: eps is smaller than alpha!

    Examples:
        >>> inputs = np.array([[0.1, 0.2, 0.6], [0.3, 0, 0.4]])
        >>> labels = np.array([[0, 1, 0, 0, 0], [0, 0, 1, 0, 0]])
        >>> attack = RandomFastGradientMethod(network, loss_fn=SoftmaxCrossEntropyWithLogits(sparse=False))
        >>> adv_x = attack.generate(inputs, labels)
    """

    def __init__(self, network, eps=0.07, alpha=0.035, bounds=(0.0, 1.0),
                 norm_level=2, is_targeted=False, loss_fn=None):
        if eps < alpha:
            raise ValueError('eps must be larger than alpha!')
        super(RandomFastGradientMethod, self).__init__(network,
                                                       eps=eps,
                                                       alpha=alpha,
                                                       bounds=bounds,
                                                       norm_level=norm_level,
                                                       is_targeted=is_targeted,
                                                       loss_fn=loss_fn)


class FastGradientSignMethod(GradientMethod):
    """
    Use the sign instead of the value of the gradient to the input. This attack is
    often referred to as Fast Gradient Sign Method and was introduced previously.

    References: `Ian J. Goodfellow, J. Shlens, and C. Szegedy, "Explaining
    and harnessing adversarial examples," in ICLR, 2015
    <https://arxiv.org/abs/1412.6572>`_

    Args:
        network (Cell): Target model.
        eps (float): Proportion of single-step adversarial perturbation generated
            by the attack to data range. Default: 0.07.
        alpha (float): Proportion of single-step random perturbation to data range.
            Default: None.
        bounds (tuple): Upper and lower bounds of data, indicating the data range.
            In form of (clip_min, clip_max). Default: (0.0, 1.0).
        is_targeted (bool): If True, targeted attack. If False, untargeted
            attack. Default: False.
        loss_fn (Loss): Loss function for optimization. If None, the input network \
            is already equipped with loss function. Default: None.

    Examples:
        >>> inputs = np.array([[0.1, 0.2, 0.6], [0.3, 0, 0.4]])
        >>> labels = np.array([[0, 1, 0, 0, 0], [0, 0, 1, 0, 0]])
        >>> attack = FastGradientSignMethod(network, loss_fn=SoftmaxCrossEntropyWithLogits(sparse=False))
        >>> adv_x = attack.generate(inputs, labels)
    """

    def __init__(self, network, eps=0.07, alpha=None, bounds=(0.0, 1.0),
                 is_targeted=False, loss_fn=None):
        super(FastGradientSignMethod, self).__init__(network,
                                                     eps=eps,
                                                     alpha=alpha,
                                                     bounds=bounds,
                                                     loss_fn=loss_fn)
        self._is_targeted = check_param_type('is_targeted', is_targeted, bool)

    def _gradient(self, inputs, labels):
        """
        Calculate gradients based on input samples and original/target
        labels.

        Args:
            inputs (Union[numpy.ndarray, tuple]): Input samples.
            labels (Union[numpy.ndarray, tuple]): original/target labels. \
                for each input if it has more than one label, it is wrapped in a tuple.

        Returns:
            numpy.ndarray, gradient of inputs.
        """
        inputs_tensor = to_tensor_tuple(inputs)
        labels_tensor = to_tensor_tuple(labels)
        out_grad = self._grad_all(*inputs_tensor, *labels_tensor)
        if isinstance(out_grad, tuple):
            out_grad = out_grad[0]
        gradient = out_grad.asnumpy()
        if self._is_targeted:
            gradient = -gradient
        gradient = np.sign(gradient)
        return gradient


class RandomFastGradientSignMethod(FastGradientSignMethod):
    """
    Fast Gradient Sign Method using random perturbation.

    References: `F. Tramer, et al., "Ensemble adversarial training: Attacks
    and defenses," in ICLR, 2018 <https://arxiv.org/abs/1705.07204>`_

    Args:
        network (Cell): Target model.
        eps (float): Proportion of single-step adversarial perturbation generated
            by the attack to data range. Default: 0.07.
        alpha (float): Proportion of single-step random perturbation to data range.
            Default: 0.035.
        bounds (tuple): Upper and lower bounds of data, indicating the data range.
            In form of (clip_min, clip_max). Default: (0.0, 1.0).
        is_targeted (bool): True: targeted attack. False: untargeted attack.
            Default: False.
        loss_fn (Loss): Loss function for optimization. If None, the input network \
            is already equipped with loss function. Default: None.

    Raises:
        ValueError: eps is smaller than alpha!

    Examples:
        >>> inputs = np.array([[0.1, 0.2, 0.6], [0.3, 0, 0.4]])
        >>> labels = np.array([[0, 1, 0, 0, 0], [0, 0, 1, 0, 0]])
        >>> attack = RandomFastGradientSignMethod(network, loss_fn=SoftmaxCrossEntropyWithLogits(sparse=False))
        >>> adv_x = attack.generate(inputs, labels)
    """

    def __init__(self, network, eps=0.07, alpha=0.035, bounds=(0.0, 1.0),
                 is_targeted=False, loss_fn=None):
        if eps < alpha:
            raise ValueError('eps must be larger than alpha!')
        super(RandomFastGradientSignMethod, self).__init__(network,
                                                           eps=eps,
                                                           alpha=alpha,
                                                           bounds=bounds,
                                                           is_targeted=is_targeted,
                                                           loss_fn=loss_fn)


class LeastLikelyClassMethod(FastGradientSignMethod):
    """
    Least-Likely Class Method.

    References: `F. Tramer, et al., "Ensemble adversarial training: Attacks
    and defenses," in ICLR, 2018 <https://arxiv.org/abs/1705.07204>`_

    Args:
        network (Cell): Target model.
        eps (float): Proportion of single-step adversarial perturbation generated
            by the attack to data range. Default: 0.07.
        alpha (float): Proportion of single-step random perturbation to data range.
            Default: None.
        bounds (tuple): Upper and lower bounds of data, indicating the data range.
            In form of (clip_min, clip_max). Default: (0.0, 1.0).
        loss_fn (Loss): Loss function for optimization. If None, the input network \
            is already equipped with loss function. Default: None.

    Examples:
        >>> inputs = np.array([[0.1, 0.2, 0.6], [0.3, 0, 0.4]])
        >>> labels = np.array([[0, 1, 0, 0, 0], [0, 0, 1, 0, 0]])
        >>> attack = LeastLikelyClassMethod(network, loss_fn=SoftmaxCrossEntropyWithLogits(sparse=False))
        >>> adv_x = attack.generate(inputs, labels)
    """

    def __init__(self, network, eps=0.07, alpha=None, bounds=(0.0, 1.0),
                 loss_fn=None):
        super(LeastLikelyClassMethod, self).__init__(network,
                                                     eps=eps,
                                                     alpha=alpha,
                                                     bounds=bounds,
                                                     is_targeted=True,
                                                     loss_fn=loss_fn)


class RandomLeastLikelyClassMethod(FastGradientSignMethod):
    """
    Least-Likely Class Method use Random perturbation.

    References: `F. Tramer, et al., "Ensemble adversarial training: Attacks
    and defenses," in ICLR, 2018 <https://arxiv.org/abs/1705.07204>`_

    Args:
        network (Cell): Target model.
        eps (float): Proportion of single-step adversarial perturbation generated
            by the attack to data range. Default: 0.07.
        alpha (float): Proportion of single-step random perturbation to data range.
            Default: 0.035.
        bounds (tuple): Upper and lower bounds of data, indicating the data range.
            In form of (clip_min, clip_max). Default: (0.0, 1.0).
        loss_fn (Loss): Loss function for optimization. If None, the input network \
            is already equipped with loss function. Default: None.

    Raises:
        ValueError: eps is smaller than alpha!

    Examples:
        >>> inputs = np.array([[0.1, 0.2, 0.6], [0.3, 0, 0.4]])
        >>> labels = np.array([[0, 1, 0, 0, 0], [0, 0, 1, 0, 0]])
        >>> attack = RandomLeastLikelyClassMethod(network, loss_fn=SoftmaxCrossEntropyWithLogits(sparse=False))
        >>> adv_x = attack.generate(inputs, labels)
    """

    def __init__(self, network, eps=0.07, alpha=0.035, bounds=(0.0, 1.0),
                 loss_fn=None):
        if eps < alpha:
            raise ValueError('eps must be larger than alpha!')
        super(RandomLeastLikelyClassMethod, self).__init__(network,
                                                           eps=eps,
                                                           alpha=alpha,
                                                           bounds=bounds,
                                                           is_targeted=True,
                                                           loss_fn=loss_fn)

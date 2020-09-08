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
Adversarial Defense.
"""
import numpy as np

from mindspore import Tensor
from mindspore.nn import Cell, SoftmaxCrossEntropyWithLogits
from mindspore.nn import WithLossCell, TrainOneStepCell
from mindspore.nn.optim.momentum import Momentum

from mindarmour.utils._check_param import check_pair_numpy_param, check_model, \
    check_param_in_range, check_param_type, check_param_multi_types
from .defense import Defense


class AdversarialDefense(Defense):
    """
    Adversarial training using given adversarial examples.

    Args:
        network (Cell): A MindSpore network to be defensed.
        loss_fn (Functions): Loss function. Default: None.
        optimizer (Cell): Optimizer used to train the network. Default: None.

    Examples:
        >>> class Net(Cell):
        >>>     def __init__(self):
        >>>         super(Net, self).__init__()
        >>>         self._reshape = P.Reshape()
        >>>         self._full_con_1 = Dense(28*28, 120)
        >>>         self._full_con_2 = Dense(120, 84)
        >>>         self._full_con_3 = Dense(84, 10)
        >>>         self._relu = ReLU()
        >>>
        >>>     def construct(self, x):
        >>>         out = self._reshape(x, (-1, 28*28))
        >>>         out = self._full_con_1(out)
        >>>         out = self.relu(out)
        >>>         out = self._full_con_2(out)
        >>>         out = self.relu(out)
        >>>         out = self._full_con_3(out)
        >>>         return out
        >>>
        >>> net = Net()
        >>> lr = 0.0001
        >>> momentum = 0.9
        >>> loss_fn = SoftmaxCrossEntropyWithLogits(sparse=True)
        >>> optimizer = Momentum(net.trainable_params(), lr, momentum)
        >>> adv_defense = AdversarialDefense(net, loss_fn, optimizer)
        >>> inputs = np.random.rand(32, 1, 28, 28).astype(np.float32)
        >>> labels = np.random.randint(0, 10).astype(np.int32)
        >>> adv_defense.defense(inputs, labels)
    """

    def __init__(self, network, loss_fn=None, optimizer=None):
        super(AdversarialDefense, self).__init__(network)
        network = check_model('network', network, Cell)
        if loss_fn is None:
            loss_fn = SoftmaxCrossEntropyWithLogits(sparse=True)

        if optimizer is None:
            optimizer = Momentum(
                params=network.trainable_params(),
                learning_rate=0.01,
                momentum=0.9)

        loss_net = WithLossCell(network, loss_fn)
        self._train_net = TrainOneStepCell(loss_net, optimizer)
        self._train_net.set_train()

    def defense(self, inputs, labels):
        """
        Enhance model via training with input samples.

        Args:
            inputs (numpy.ndarray): Input samples.
            labels (numpy.ndarray): Labels of input samples.

        Returns:
            numpy.ndarray, loss of defense operation.
        """
        inputs, labels = check_pair_numpy_param('inputs', inputs, 'labels',
                                                labels)
        loss = self._train_net(Tensor(inputs), Tensor(labels))
        return loss.asnumpy()


class AdversarialDefenseWithAttacks(AdversarialDefense):
    """
    Adversarial defense with attacks.

    Args:
        network (Cell): A MindSpore network to be defensed.
        attacks (list[Attack]): List of attack method.
        loss_fn (Functions): Loss function. Default: None.
        optimizer (Cell): Optimizer used to train the network. Default: None.
        bounds (tuple): Upper and lower bounds of data. In form of (clip_min,
            clip_max). Default: (0.0, 1.0).
        replace_ratio (float): Ratio of replacing original samples with
            adversarial, which must be between 0 and 1. Default: 0.5.

    Raises:
        ValueError: If replace_ratio is not between 0 and 1.

    Examples:
        >>> net = Net()
        >>> fgsm = FastGradientSignMethod(net)
        >>> pgd = ProjectedGradientDescent(net)
        >>> ead = AdversarialDefenseWithAttacks(net, [fgsm, pgd])
        >>> ead.defense(inputs, labels)
    """

    def __init__(self, network, attacks, loss_fn=None, optimizer=None,
                 bounds=(0.0, 1.0), replace_ratio=0.5):
        super(AdversarialDefenseWithAttacks, self).__init__(network,
                                                            loss_fn,
                                                            optimizer)
        self._attacks = check_param_type('attacks', attacks, list)
        self._bounds = check_param_multi_types('bounds', bounds, [tuple, list])
        for elem in self._bounds:
            _ = check_param_multi_types('bound', elem, [int, float])
        self._replace_ratio = check_param_in_range('replace_ratio',
                                                   replace_ratio,
                                                   0, 1)
        self._graph_initialized = False

    def defense(self, inputs, labels):
        """
        Enhance model via training with adversarial examples generated from input samples.

        Args:
            inputs (numpy.ndarray): Input samples.
            labels (numpy.ndarray): Labels of input samples.

        Returns:
            numpy.ndarray, loss of adversarial defense operation.
        """
        inputs, labels = check_pair_numpy_param('inputs', inputs, 'labels',
                                                labels)
        if not self._graph_initialized:
            self._train_net(Tensor(inputs), Tensor(labels))
            self._graph_initialized = True

        x_len = inputs.shape[0]
        n_adv = int(np.ceil(self._replace_ratio*x_len))
        n_adv_per_attack = int(n_adv / len(self._attacks))

        adv_ids = np.random.choice(x_len, size=n_adv, replace=False)
        start = 0
        for attack in self._attacks:
            idx = adv_ids[start:start + n_adv_per_attack]
            inputs[idx] = attack.generate(inputs[idx], labels[idx])
            start += n_adv_per_attack

        loss = self._train_net(Tensor(inputs), Tensor(labels))
        return loss.asnumpy()


class EnsembleAdversarialDefense(AdversarialDefenseWithAttacks):
    """
    Ensemble adversarial defense.

    Args:
        network (Cell): A MindSpore network to be defensed.
        attacks (list[Attack]): List of attack method.
        loss_fn (Functions): Loss function. Default: None.
        optimizer (Cell): Optimizer used to train the network. Default: None.
        bounds (tuple): Upper and lower bounds of data. In form of (clip_min,
            clip_max). Default: (0.0, 1.0).
        replace_ratio (float): Ratio of replacing original samples with
            adversarial, which must be between 0 and 1. Default: 0.5.

    Raises:
        ValueError: If replace_ratio is not between 0 and 1.

    Examples:
        >>> net = Net()
        >>> fgsm = FastGradientSignMethod(net)
        >>> pgd = ProjectedGradientDescent(net)
        >>> ead = EnsembleAdversarialDefense(net, [fgsm, pgd])
        >>> ead.defense(inputs, labels)
    """

    def __init__(self, network, attacks, loss_fn=None, optimizer=None,
                 bounds=(0.0, 1.0), replace_ratio=0.5):
        super(EnsembleAdversarialDefense, self).__init__(network,
                                                         attacks,
                                                         loss_fn,
                                                         optimizer,
                                                         bounds,
                                                         replace_ratio)

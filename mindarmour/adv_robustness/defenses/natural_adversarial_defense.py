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
Natural Adversarial Defense.
"""
from ..attacks.gradient_method import FastGradientSignMethod
from .adversarial_defense import AdversarialDefenseWithAttacks


class NaturalAdversarialDefense(AdversarialDefenseWithAttacks):
    """
    Adversarial training based on FGSM.

    Reference: `A. Kurakin, et al., "Adversarial machine learning at scale," in
    ICLR, 2017. <https://arxiv.org/abs/1611.01236>`_

    Args:
        network (Cell): A MindSpore network to be defensed.
        loss_fn (Functions): Loss function. Default: None.
        optimizer (Cell): Optimizer used to train the network. Default: None.
        bounds (tuple): Upper and lower bounds of data. In form of (clip_min,
            clip_max). Default: (0.0, 1.0).
        replace_ratio (float): Ratio of replacing original samples with
            adversarial samples. Default: 0.5.
        eps (float): Step size of the attack method(FGSM). Default: 0.1.

    Examples:
        >>> net = Net()
        >>> adv_defense = NaturalAdversarialDefense(net)
        >>> adv_defense.defense(inputs, labels)
    """
    def __init__(self, network, loss_fn=None, optimizer=None,
                 bounds=(0.0, 1.0), replace_ratio=0.5, eps=0.1):
        attack = FastGradientSignMethod(network,
                                        eps=eps,
                                        alpha=None,
                                        bounds=bounds,
                                        loss_fn=loss_fn)
        super(NaturalAdversarialDefense, self).__init__(
            network,
            [attack],
            loss_fn=loss_fn,
            optimizer=optimizer,
            bounds=bounds,
            replace_ratio=replace_ratio)

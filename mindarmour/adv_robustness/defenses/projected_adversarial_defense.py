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
Projected Adversarial Defense.
"""
from ..attacks.iterative_gradient_method import ProjectedGradientDescent
from .adversarial_defense import AdversarialDefenseWithAttacks


class ProjectedAdversarialDefense(AdversarialDefenseWithAttacks):
    """
    Adversarial training based on PGD.

    Reference: `A. Madry, et al., "Towards deep learning models resistant to
    adversarial attacks," in ICLR, 2018. <https://arxiv.org/abs/1611.01236>`_

    Args:
        network (Cell): A MindSpore network to be defensed.
        loss_fn (Functions): Loss function. Default: None.
        optimizer (Cell): Optimizer used to train the nerwork. Default: None.
        bounds (tuple): Upper and lower bounds of input data. In form of
            (clip_min, clip_max). Default: (0.0, 1.0).
        replace_ratio (float): Ratio of replacing original samples with
            adversarial samples. Default: 0.5.
        eps (float): PGD attack parameters, epsilon. Default: 0.3.
        eps_iter (int): PGD attack parameters, inner loop epsilon.
            Default:0.1.
        nb_iter (int): PGD attack parameters, number of iteration.
            Default: 5.
        norm_level (str): Norm type. 'inf' or 'l2'. Default: 'inf'.

    Examples:
        >>> net = Net()
        >>> adv_defense = ProjectedAdversarialDefense(net)
        >>> adv_defense.defense(inputs, labels)
    """
    def __init__(self,
                 network,
                 loss_fn=None,
                 optimizer=None,
                 bounds=(0.0, 1.0),
                 replace_ratio=0.5,
                 eps=0.3,
                 eps_iter=0.1,
                 nb_iter=5,
                 norm_level='inf'):
        attack = ProjectedGradientDescent(network,
                                          eps=eps,
                                          eps_iter=eps_iter,
                                          nb_iter=nb_iter,
                                          bounds=bounds,
                                          norm_level=norm_level,
                                          loss_fn=loss_fn)
        super(ProjectedAdversarialDefense, self).__init__(
            network, [attack], loss_fn=loss_fn, optimizer=optimizer,
            bounds=bounds, replace_ratio=replace_ratio)

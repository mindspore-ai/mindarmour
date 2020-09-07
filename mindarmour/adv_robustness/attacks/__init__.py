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
This module includes classical black-box and white-box attack algorithms
in making adversarial examples.
"""
from .gradient_method import *
from .iterative_gradient_method import *
from .deep_fool import DeepFool
from .jsma import JSMAAttack
from .carlini_wagner import CarliniWagnerL2Attack
from .lbfgs import LBFGS
from . import black
from .black.hop_skip_jump_attack import HopSkipJumpAttack
from .black.genetic_attack import GeneticAttack
from .black.natural_evolutionary_strategy import NES
from .black.pointwise_attack import PointWiseAttack
from .black.pso_attack import PSOAttack
from .black.salt_and_pepper_attack import SaltAndPepperNoiseAttack

__all__ = ['FastGradientMethod',
           'RandomFastGradientMethod',
           'FastGradientSignMethod',
           'RandomFastGradientSignMethod',
           'LeastLikelyClassMethod',
           'RandomLeastLikelyClassMethod',
           'IterativeGradientMethod',
           'BasicIterativeMethod',
           'MomentumIterativeMethod',
           'ProjectedGradientDescent',
           'DiverseInputIterativeMethod',
           'MomentumDiverseInputIterativeMethod',
           'DeepFool',
           'CarliniWagnerL2Attack',
           'JSMAAttack',
           'LBFGS',
           'GeneticAttack',
           'HopSkipJumpAttack',
           'NES',
           'PointWiseAttack',
           'PSOAttack',
           'SaltAndPepperNoiseAttack'
           ]

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

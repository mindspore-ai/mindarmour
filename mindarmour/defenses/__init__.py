"""
This module includes classical defense algorithms in defencing adversarial
examples and enhancing model security and trustworthy.
"""
from .adversarial_defense import AdversarialDefense
from .adversarial_defense import AdversarialDefenseWithAttacks
from .adversarial_defense import EnsembleAdversarialDefense
from .natural_adversarial_defense import NaturalAdversarialDefense
from .projected_adversarial_defense import ProjectedAdversarialDefense

__all__ = ['AdversarialDefense',
           'AdversarialDefenseWithAttacks',
           'NaturalAdversarialDefense',
           'ProjectedAdversarialDefense',
           'EnsembleAdversarialDefense']

"""
This module provides implementations of inference tampering attacks and defenses.
It includes BadNets and Blended attacks, as well as the STRIP defense mechanism.
"""

from .attack import BadNets, Blended
from .strip import STRIP

__all__ = ['BadNets', 'Blended', 'STRIP']

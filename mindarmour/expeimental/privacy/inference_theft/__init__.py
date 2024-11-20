'''
This module provides implementations of inference theft attacks and defenses.
KnockOff:
Anti-theft during inference phase, supports PyTorch, MindSpore, PaddlePaddle,
TensorFlow. Applicable to general and traffic scenarios. General scenarios
include CIFAR, SVHN, FashionMNIST; traffic scenarios include GTSRB. This
algorithm improves the efficiency of theft attacks by sampling from public
datasets.

MAZE:
Anti-theft during inference phase, supports PyTorch. Applicable to general and
traffic scenarios. General scenarios include CIFAR, SVHN, FashionMNIST; traffic
scenarios include GTSRB. This algorithm uses generative models to compensate
for the lack of training set knowledge, improving the efficiency of theft
attacks.

PVMTA:
Anti-theft during inference phase, supports PyTorch, MindSpore, PaddlePaddle,
TensorFlow. Applicable to general and traffic scenarios. General scenarios
include CIFAR, SVHN, FashionMNIST; traffic scenarios include GTSRB. This
algorithm dynamically adjusts the temperature of SoftMax using confidence to
reduce information, significantly reducing the efficiency of theft attacks.
'''

from .knockoff import Knockoff
from .pvmta import PVMTA
from .maze import MAZE

__all__ = ['Knockoff', 'PVMTA', 'MAZE']

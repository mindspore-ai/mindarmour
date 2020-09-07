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

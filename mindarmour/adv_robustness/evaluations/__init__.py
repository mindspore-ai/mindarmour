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
This module includes various metrics to evaluate the result of attacks or
defenses.
"""
from .attack_evaluation import AttackEvaluate
from .defense_evaluation import DefenseEvaluate
from .visual_metrics import RadarMetric
from . import black
from .black.defense_evaluation import BlackDefenseEvaluate

__all__ = ['AttackEvaluate',
           'BlackDefenseEvaluate',
           'DefenseEvaluate',
           'RadarMetric']

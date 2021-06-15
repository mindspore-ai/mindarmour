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
MindArmour, a tool box of MindSpore to enhance model trustworthiness and achieve
privacy-preserving machine learning.
"""
from .adv_robustness.attacks import Attack
from .adv_robustness.attacks.black.black_model import BlackModel
from .adv_robustness.defenses.defense import Defense
from .adv_robustness.detectors.detector import Detector
from .fuzz_testing.fuzzing import Fuzzer
from .privacy.diff_privacy import DPModel
from .privacy.evaluation.membership_inference import MembershipInference
from .privacy.sup_privacy.sup_ctrl.conctrl import SuppressCtrl
from .privacy.sup_privacy.train.model import SuppressModel
from .privacy.sup_privacy.mask_monitor.masker import SuppressMasker
from .privacy.evaluation.inversion_attack import ImageInversionAttack
from .reliability.concept_drift.concept_drift_check_time_series import ConceptDriftCheckTimeSeries

__all__ = ['Attack',
           'BlackModel',
           'Detector',
           'Defense',
           'Fuzzer',
           'DPModel',
           'MembershipInference',
           'SuppressModel',
           'SuppressCtrl',
           'SuppressMasker',
           'ImageInversionAttack',
           'ConceptDriftCheckTimeSeries']

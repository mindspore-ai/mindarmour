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
import logging
import time

from .adv_robustness.attacks.attack import Attack
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


def _mindspore_version_check():
    """
    Do the MindSpore version check for MindArmour. If the
    MindSpore can not be imported, it will raise ImportError. If its
    version is not compatible with current MindArmour version,
    it will print a warning.

    Raise:
        ImportError: If the MindSpore can not be imported.
    """
    try:
        from mindarmour.version import __version__
        ma_version = __version__[:3]
    except (ImportError, ModuleNotFoundError):
        raise ImportError(f"Get MindArmour version failed")

    try:
        import mindspore as ms
    except (ImportError, ModuleNotFoundError):
        raise ImportError("Can not find MindSpore in current environment. Please install MindSpore before using " \
                          "MindArmour, by following the instruction at https://www.mindspore.cn/install")

    ms_ma_version_match = {'1.7': ['1.7'],
                           '1.8': ['1.7', '1.8', '1.9', '2.0'],
                           '1.9': ['1.7', '1.8', '1.9', '2.0'],
                           '2.0': ['1.7', '1.8', '1.9', '2.0']}

    ms_version = ms.__version__[:3]
    required_mindspore_verision = ms_ma_version_match.get(ma_version[:3])

    if ms_version not in required_mindspore_verision:
        logging.warning("Current version of MindSpore is not compatible with MindArmour. "
                        "Some functions might not work or even raise error. Please install MindSpore "
                        "version in %s. For more details about dependency setting, please check "
                        "the instructions at MindSpore official website https://www.mindspore.cn/install "
                        "or check the README.md at https://gitee.com/mindspore/mindarmour", required_mindspore_verision)
        warning_countdown = 3
        for i in range(warning_countdown, 0, -1):
            logging.warning("Please pay attention to the above warning, countdown: %d", i)
        time.sleep(1)

_mindspore_version_check()

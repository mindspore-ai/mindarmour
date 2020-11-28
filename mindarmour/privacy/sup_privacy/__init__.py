# Copyright 2021 Huawei Technologies Co., Ltd
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
This module provides Suppress Privacy feature to protect user privacy.
"""
from .mask_monitor.masker import SuppressMasker
from .train.model import SuppressModel
from .sup_ctrl.conctrl import SuppressPrivacyFactory
from .sup_ctrl.conctrl import SuppressCtrl
from .sup_ctrl.conctrl import MaskLayerDes

__all__ = ['SuppressMasker',
           'SuppressModel',
           'SuppressPrivacyFactory',
           'SuppressCtrl',
           'MaskLayerDes']

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
This module includes detector methods on distinguishing adversarial examples
from benign examples.
"""
from .mag_net import ErrorBasedDetector
from .mag_net import DivergenceBasedDetector
from .ensemble_detector import EnsembleDetector
from .region_based_detector import RegionBasedDetector
from .spatial_smoothing import SpatialSmoothing
from . import black
from .black.similarity_detector import SimilarityDetector

__all__ = ['ErrorBasedDetector',
           'DivergenceBasedDetector',
           'RegionBasedDetector',
           'SpatialSmoothing',
           'EnsembleDetector',
           'SimilarityDetector']

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

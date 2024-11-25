"""
This module initializes the vec2text package, providing access to various components
for embedding inversion tasks, including data collation, model training, and hypothesis generation.
"""
# pylint: disable=W0406
from . import (  # noqa: F401
    aliases,
    collator,
    metrics,
    models,
    trainers,
)
from .trainers import Corrector  # noqa: F401

"""
Robust Vision: Production-Ready Scalable Training Framework

A comprehensive framework for training robust vision models with advanced
techniques including EMA, label smoothing, mixup, and multi-GPU support.

Repository: https://github.com/or4k2l/robust-vision
"""

__version__ = "1.0.0"
__author__ = "Yahya Akbay"
__repo__ = "https://github.com/or4k2l/robust-vision"

from . import data
from . import models
from . import training
from . import evaluation
from . import utils

__all__ = [
    "data",
    "models",
    "training",
    "evaluation",
    "utils",
]

"""
This module is used to evaluate the incoming dataset with classical baselines that the user can easily modify and use interactively.
"""

from deephyper.baseline.base import BasePipeline
from deephyper.baseline.classifier import BaseClassifierPipeline
from deephyper.baseline.regressor import BaseRegressorPipeline

__all__ = ["BasePipeline", "BaseClassifierPipeline", "BaseRegressorPipeline"]

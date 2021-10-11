"""The ``ensemble`` module provides a way to build ensembles of checkpointed deep neural networks from ``tensorflow.keras``, with ``.h5`` format, to regularize and boost predictive performance as well as estimate better uncertainties.
"""
from deephyper.ensemble.base_ensemble import BaseEnsemble
from deephyper.ensemble.bagging_ensemble import (
    BaggingEnsembleRegressor,
    BaggingEnsembleClassifier,
)
from deephyper.ensemble.uq_bagging_ensemble import UQBaggingEnsembleRegressor, UQBaggingEnsembleClassifier

__all__ = [
    "BaseEnsemble",
    "BaggingEnsembleRegressor",
    "BaggingEnsembleClassifier",
    "UQBaggingEnsembleRegressor",
    "UQBaggingEnsembleClassifier",
]

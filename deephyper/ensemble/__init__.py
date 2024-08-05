"""The ``ensemble`` module provides a way to build ensembles of checkpointed deep neural networks from ``tensorflow.keras``, with ``.h5`` format, to regularize and boost predictive performance as well as estimate better uncertainties.
"""

from deephyper.ensemble._ensemble import Ensemble

from deephyper.ensemble._uq_ensemble import (
    UQEnsembleRegressor,
    UQEnsembleClassifier,
)

__all__ = [
    "Ensemble",
    "UQEnsembleRegressor",
    "UQEnsembleClassifier",
]

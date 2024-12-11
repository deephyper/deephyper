"""Predictive model subpackage."""

from deephyper.predictor._predictor import (
    Predictor,
    PredictorLoader,
    PredictorFileLoader,
)

__all__ = ["Predictor", "PredictorLoader", "PredictorFileLoader"]

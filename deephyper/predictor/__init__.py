"""Subpackage for predictor models."""

from deephyper.predictor._predictor import (
    Predictor,
    PredictorLoader,
    PredictorFileLoader,
)

__all__ = ["Predictor", "PredictorLoader", "PredictorFileLoader"]

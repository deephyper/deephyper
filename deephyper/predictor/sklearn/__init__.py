"""Subpackage for the Scikit-Learn based predictors."""

from deephyper.predictor.sklearn._predictor_sklearn import (
    SklearnPredictor,
    SklearnPredictorFileLoader,
)

__all__ = ["SklearnPredictor", "SklearnPredictorFileLoader"]

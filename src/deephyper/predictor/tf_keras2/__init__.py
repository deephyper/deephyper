"""Subpackage for the TensorFlow Keras 2 based predictors."""

from deephyper.predictor.tf_keras2._predictor_tf_keras2 import (
    TFKeras2Predictor,
    TFKeras2PredictorFileLoader,
)

__all__ = ["TFKeras2Predictor", "TFKeras2PredictorFileLoader"]

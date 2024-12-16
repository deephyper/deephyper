"""Subpackage for the PyTorch based predictors."""

from deephyper.predictor.torch._predictor_torch import (
    TorchPredictor,
    TorchPredictorFileLoader,
)

__all__ = ["TorchPredictor", "TorchPredictorFileLoader"]

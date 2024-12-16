"""Ensemble subpackage.

This subpackage provides tools to build ensemble of predictive models. Ensembling can be useful to
minimize the loss if the learner has variance (i.e., its performance metric varies when trained
multiple times) or to estimate the uncertainty of the predictions.
"""

from deephyper.ensemble._ensemble import EnsemblePredictor

__all__ = ["EnsemblePredictor"]

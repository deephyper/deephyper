"""Subpackage with aggregation functions for the predictions of set of predicting models."""

from deephyper.ensemble.aggregator._aggregator import Aggregator
from deephyper.ensemble.aggregator._mean import MeanAggregator
from deephyper.ensemble.aggregator._mixed_normal import MixedNormalAggregator
from deephyper.ensemble.aggregator._mode import ModeAggregator
from deephyper.ensemble.aggregator._mixed_categorical import MixedCategoricalAggregator

__all__ = [
    "Aggregator",
    "MeanAggregator",
    "MixedNormalAggregator",
    "ModeAggregator",
    "MixedCategoricalAggregator",
]

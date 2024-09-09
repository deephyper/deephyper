"""Subpackage providing methods that select a subset of predictors from a set of available predictors in order to build an ensemble."""

from deephyper.ensemble.selector._selector import Selector
from deephyper.ensemble.selector._topk import TopKSelector
from deephyper.ensemble.selector._greedy import GreedySelector


__all__ = ["Selector", "TopKSelector", "GreedySelector"]

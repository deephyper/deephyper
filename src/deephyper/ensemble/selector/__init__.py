"""Subpackage providing ensemble selection algorithms.

It contains methods that select a subset of predictors from a set of available predictors
in order to build an ensemble.
"""

from deephyper.ensemble.selector._greedy import GreedySelector
from deephyper.ensemble.selector._online_selector import OnlineSelector
from deephyper.ensemble.selector._selector import Selector
from deephyper.ensemble.selector._topk import TopKSelector


__all__ = ["GreedySelector", "OnlineSelector", "Selector", "TopKSelector"]

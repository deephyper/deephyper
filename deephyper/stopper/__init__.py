"""The ``stopper`` module provides features to observe intermediate performances of a black-box function and allow to stop or continue its evaluation with respect to some budget.

This module was inspired from the Pruner interface and implementation of `Optuna <https://optuna.readthedocs.io/en/stable/reference/pruners.html>`_.
"""

from deephyper.stopper._stopper import Stopper
from deephyper.stopper._asha_stopper import SuccessiveHalvingStopper
from deephyper.stopper._median_stopper import MedianStopper

__all__ = ["Stopper", "SuccessiveHalvingStopper", "MedianStopper"]

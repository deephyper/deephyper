"""The ``stopper`` module provides features to observe intermediate performances of a black-box function and allow to stop or continue its evaluation with respect to some budget.

This module was inspired from the Pruner interface and implementation of `Optuna <https://optuna.readthedocs.io/en/stable/reference/pruners.html>`_.
"""

from deephyper.stopper._stopper import Stopper
from deephyper.stopper._asha_stopper import SuccessiveHalvingStopper
from deephyper.stopper._median_stopper import MedianStopper
from deephyper.stopper._idle_stopper import IdleStopper
from deephyper.stopper._const_stopper import ConstantStopper


__all__ = [
    "IdleStopper",
    "Stopper",
    "SuccessiveHalvingStopper",
    "MedianStopper",
    "ConstantStopper",
]


try:

    from deephyper.stopper._lcmodel_stopper import LCModelStopper  # noqa: F401

    __all__.append("LCModelStopper")
except ImportError:
    pass

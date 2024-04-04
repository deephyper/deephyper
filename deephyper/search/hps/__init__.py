"""Sub-package for hyperparameter search algorithms.

.. warning:: All search algorithms are MAXIMIZING the objective function. If you want to MINIMIZE the objective function, you have to return the negative of you objective.
"""

from deephyper.search.hps._cbo import CBO, AMBS
from deephyper.search.hps._eds import ExperimentalDesignSearch

__all__ = ["CBO", "AMBS", "ExperimentalDesignSearch"]

try:
    from deephyper.search.hps._mpi_dbo import MPIDistributedBO  # noqa: F401

    __all__.append("MPIDistributedBO")
except ImportError:
    pass

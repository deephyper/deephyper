"""Subpackage for hyperparameter search algorithms.

.. warning:: All search algorithms are MAXIMIZING the objective function. If you want to MINIMIZE the objective function, you have to return the negative of you objective.
"""

from deephyper.hpo._search import Search
from deephyper.hpo._cbo import CBO
from deephyper.hpo._eds import ExperimentalDesignSearch

__all__ = ["CBO", "ExperimentalDesignSearch", "Search"]

try:
    from deephyper.hpo._mpi_dbo import MPIDistributedBO  # noqa: F401

    __all__.append("MPIDistributedBO")
except ImportError:
    pass

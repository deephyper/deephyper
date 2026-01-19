"""Hyperparameter optimization and neural architecture search subpackage.

.. warning::

   All optimization algorithms are MAXIMIZING the objective function. If you want to
   MINIMIZE instead, you should remember to return the negative of your objective.
"""

from deephyper.hpo._cbo import CBO
from deephyper.hpo._eds import ExperimentalDesignSearch
from deephyper.hpo._problem import HpProblem
from deephyper.hpo._random import RandomSearch
from deephyper.hpo._regevo import RegularizedEvolution
from deephyper.hpo._search import Search
from deephyper.hpo.solution import (
    ArgMaxEstSelection,
    ArgMaxObsSelection,
    Solution,
    SolutionSelection,
)

__all__ = [
    # Definition of hyperparameter space
    "HpProblem",
    # Optimization algorithms
    "CBO",
    "ExperimentalDesignSearch",
    "RandomSearch",
    "RegularizedEvolution",
    "Search",
    # Solution selection
    "ArgMaxEstSelection",
    "ArgMaxObsSelection",
    "Solution",
    "SolutionSelection",
]

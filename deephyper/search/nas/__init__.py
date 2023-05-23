"""Sub-package for neural architecture search algorithms.

.. warning:: All search algorithms are MAXIMIZING the objective function. If you want to MINIMIZE the objective function, you have to return the negative of you objective.
"""
from deephyper.search.nas._base import NeuralArchitectureSearch
from deephyper.search.nas._regevo import RegularizedEvolution
from deephyper.search.nas._agebo import AgEBO
from deephyper.search.nas._ambsmixed import AMBSMixed
from deephyper.search.nas._random import Random
from deephyper.search.nas._regevomixed import RegularizedEvolutionMixed

__all__ = [
    "AgEBO",
    "AMBSMixed",
    "NeuralArchitectureSearch",
    "Random",
    "RegularizedEvolution",
    "RegularizedEvolutionMixed",
]

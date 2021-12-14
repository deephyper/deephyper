"""Neural architecture search algorithms.
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

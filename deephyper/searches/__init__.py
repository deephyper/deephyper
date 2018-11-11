"""
The ``searches`` module bring a modular way to implement new search algorithms and two sub modules. One is for hyperparameter search ``deephyper.searches.hps`` and one is for neural architecture search ``deephyper.searches.nas``.
"""
from deephyper.searches.search import Search

__all__ = ['Search']

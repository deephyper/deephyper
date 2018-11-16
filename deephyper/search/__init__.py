"""
The ``search`` module bring a modular way to implement new search algorithms and two sub modules. One is for hyperparameter search ``deephyper.search.hps`` and one is for neural architecture search ``deephyper.search.nas``.
"""
from deephyper.search.search import Search

__all__ = ['Search']

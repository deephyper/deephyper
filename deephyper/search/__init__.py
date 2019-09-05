"""
The ``search`` module brings a modular way to implement new search algorithms and two sub modules. One is for hyperparameter search ``deephyper.search.hps`` and one is for neural architecture search ``deephyper.search.nas``.
The ``Search`` class is abstract and has different subclasses such as: ``deephyper.search.ambs`` and ``deephyper.search.ga``.
"""

from deephyper.search.search import Search

__all__ = [
    'Search',
    'nas',
    'hps'
    ]

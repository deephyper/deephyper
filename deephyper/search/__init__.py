"""This subpackage provides an interface to implement new search algorithms as well as some already implemented search algorithms. One module of this subpackage is specialized for hyperparameter optimization algorithms ``deephyper.search.hps`` and an other is specialized for neural architecture search ``deephyper.search.nas``.

The :class:`deephyper.search.Search` class provides the generic interface of a search.

.. warning:: All search algorithms are MAXIMIZING the objective function. If you want to MINIMIZE the objective function, you have to return the negative of you objective.
"""

from deephyper.search._search import Search

__all__ = ["Search"]

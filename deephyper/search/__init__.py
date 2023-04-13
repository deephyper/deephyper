"""This sub-package provides an interface to implement new search algorithms as well as some already implemented search algorithms. One module of this sub-package is specialized for hyperparameter optimization algorithms ``deephyper.search.hps`` and an other is specialized for neural architecture search ``deephyper.search.nas``.

The :class:`deephyper.search.Search` class provides the generic interface of a search.
"""

from deephyper.search._search import Search

__all__ = ["Search"]

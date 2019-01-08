"""
The ``search`` module brings a modular way to implement new search algorithms. The ``Search`` class is abstract and has two subclasses ``deephyper.search.ambs`` and ``deephyper.search.ga``.
"""

from deephyper.search.search import Search
from deephyper.search.ambs import AMBS
from deephyper.search.ga import GA

__all__ = ['Search', 'AMBS']

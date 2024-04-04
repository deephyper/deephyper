"""This subpackage his dedicated to the definition of neural architecture search space and evaluation strategy. The implementation is using Tensorflow 2.X and Keras API. The main concepts are:
* :class:`deephyper.nas.KSearchSpace`: An object to define a search space of neural architectures.
* :mod:`deephyper.nas.run`: A subpackage to define the evaluation strategy of a neural architecture (e.g., training procedure).
* :mod:`deephyper.nas.operation`: A subpackage to define operations of the neural architecture search space.
* :mod:`deephyper.nas.node`: A subpackage to define nodes of the neural architecture search space which is represented as a direct acyclic graph.
"""

from ._nx_search_space import NxSearchSpace
from ._keras_search_space import KSearchSpace

__all__ = ["NxSearchSpace", "KSearchSpace"]

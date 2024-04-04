"""This subpackage provides tools to define hyperparameter and neural architecture search problems. Some features of this module are based on the `ConfigSpace <https://automl.github.io/ConfigSpace/master/>`_ project. The main classes provided by this module are:

- :class:`deephyper.problem.HpProblem`: A class to define a hyperparameter search problem.
- :class:`deephyper.problem.NaProblem`: A class to define a neural architecture search problem.
"""

from ConfigSpace import *  # noqa: F401, F403

from ._hyperparameter import HpProblem

__all__ = ["HpProblem"]

# make import of NaProblem optional
try:
    from ._neuralarchitecture import NaProblem  # noqa: F401

    __all__.append("NaProblem")
except ModuleNotFoundError as e:
    if "tensorflow" in str(e):
        pass
    elif "networkx" in str(e):
        pass
    else:
        raise e

__all__ = ["HpProblem"]

"""This module provides tools to define hyperparameter and neural architecture search problems. Some features of this module are based on the `ConfigSpace <https://automl.github.io/ConfigSpace/master/>`_ project.
"""
from ConfigSpace import *

from ._hyperparameter import HpProblem

__all__ = ["HpProblem"]

try:
    from ._neuralarchitecture import NaProblem
    __all__.append("NaProblem")
except ModuleNotFoundError as e:
    if not("tensorflow" in str(e)):
        raise e

__all__ = ["HpProblem"]

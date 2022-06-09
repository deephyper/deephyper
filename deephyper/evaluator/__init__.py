"""
This evaluator module asynchronously manages a series of Job objects to help execute given HPS or NAS tasks on various environments with differing system settings and properties.
"""

from deephyper.evaluator._evaluator import EVALUATORS, Evaluator
from deephyper.evaluator._job import Job
from deephyper.evaluator._process_pool import ProcessPoolEvaluator
from deephyper.evaluator._serial import SerialEvaluator
from deephyper.evaluator._subprocess import SubprocessEvaluator
from deephyper.evaluator._thread_pool import ThreadPoolEvaluator
from deephyper.evaluator._queue import queued
from deephyper.evaluator._decorator import profile

__all__ = [
    "Evaluator",
    "EVALUATORS",
    "Job",
    "ProcessPoolEvaluator",
    "profile",
    "queued",
    "SerialEvaluator",
    "SubprocessEvaluator",
    "ThreadPoolEvaluator",
]

try:

    from deephyper.evaluator._ray import RayEvaluator

    __all__.append("RayEvaluator")
except ImportError:
    pass

try:
    from deephyper.evaluator._mpi_pool import MPIPoolEvaluator

    __all__.append("MPIPoolEvaluator")
except ImportError:
    pass

try:
    from deephyper.evaluator._mpi_comm import MPICommEvaluator

    __all__.append("MPICommEvaluator")
except ImportError:
    pass

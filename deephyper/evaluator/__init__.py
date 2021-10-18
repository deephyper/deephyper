"""
This evaluator module asynchronously manages a series of Job objects to help execute given HPS or NAS tasks on various environments with differing system settings and properties.
"""

from deephyper.evaluator._evaluator import EVALUATORS, Evaluator
from deephyper.evaluator._job import Job
from deephyper.evaluator._process_pool import ProcessPoolEvaluator
from deephyper.evaluator._ray import RayEvaluator
from deephyper.evaluator._subprocess import SubprocessEvaluator
from deephyper.evaluator._thread_pool import ThreadPoolEvaluator

__all__ = [
    "EVALUATORS",
    "Evaluator",
    "Job",
    "ProcessPoolEvaluator",
    "RayEvaluator",
    "SubprocessEvaluator",
    "ThreadPoolEvaluator"
]

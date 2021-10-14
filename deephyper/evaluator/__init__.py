"""
This evaluator module asynchronously manages a series of Job objects to help execute given HPS or NAS tasks on various environments with differing system settings and properties.
"""

from deephyper.evaluator._evaluator import Evaluator, EVALUATORS
from deephyper.evaluator._job import Job

__all__ = [
    "Evaluator",
    "Job",
    "EVALUATORS",
]

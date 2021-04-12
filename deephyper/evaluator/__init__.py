"""
The goal off the evaluator module is to have a set of objects which can helps us to run our task on different environments and with different system settings/properties.
"""

from deephyper.evaluator.evaluate import Encoder
from deephyper.evaluator._balsam import BalsamEvaluator
from deephyper.evaluator._ray_evaluator import RayEvaluator
from deephyper.evaluator._subprocess import SubprocessEvaluator

__all__ = ["Encoder", "BalsamEvaluator", "RayEvaluator", "SubprocessEvaluator"]

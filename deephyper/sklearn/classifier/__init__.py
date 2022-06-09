from deephyper.sklearn.classifier._autosklearn1 import (
    problem_autosklearn1,
    run_autosklearn1,
)

__all__ = ["problem_autosklearn1", "run_autosklearn1"]

__doc__ = """
AutoML searches are executed with the ``deephyper.search.hps.CBO`` algorithm only. We provide ready to go problems, and run functions for you to use it easily.
"""

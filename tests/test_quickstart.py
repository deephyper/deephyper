import unittest

import pytest


def run(config: dict):
    return -config["x"] ** 2


@pytest.mark.hps_fast_test
class QuickStartTest(unittest.TestCase):
    def test_quickstart(self):
        from deephyper.problem import HpProblem
        from deephyper.search.hps import CBO
        from deephyper.evaluator import Evaluator

        # define the variable you want to optimize
        problem = HpProblem()
        problem.add_hyperparameter((-10.0, 10.0), "x")

        # define the evaluator to distribute the computation
        evaluator = Evaluator.create(
            run,
            method="subprocess",
            method_kwargs={
                "num_workers": 2,
            },
        )

        # define you search and execute it
        search = CBO(problem, evaluator)

        results = search.search(max_evals=15)

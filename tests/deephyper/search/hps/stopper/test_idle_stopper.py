import unittest
import pytest

import numpy as np

from deephyper.evaluator import RunningJob
from deephyper.problem import HpProblem
from deephyper.search.hps import CBO
from deephyper.stopper import IdleStopper


def run(job: RunningJob) -> dict:

    assert isinstance(job.stopper, IdleStopper)

    max_budget = 50
    objective_i = 0

    for budget_i in range(1, max_budget + 1):
        objective_i += job["x"]

        job.record(budget_i, objective_i)
        if job.stopped():
            break

    return {
        "objective": job.observations,
        "metadata": {"budget": budget_i, "stopped": budget_i < max_budget},
    }


@pytest.mark.fast
@pytest.mark.hps
class TestSHAStopper(unittest.TestCase):
    def test_basic(self):

        # define the variable you want to optimize
        problem = HpProblem()
        problem.add_hyperparameter((0.0, 10.0), "x")

        stopper = IdleStopper()
        search = CBO(
            problem, run, surrogate_model="DUMMY", stopper=stopper, random_state=42
        )

        results = search.search(max_evals=30)

        assert "m:budget" in results.columns
        assert "m:stopped" in results.columns
        assert "p:x" in results.columns
        assert "objective" in results.columns

        budgets = np.sort(np.unique(results["m:budget"].to_numpy())).tolist()
        assert budgets == [50]


if __name__ == "__main__":
    test = TestSHAStopper()
    test.test_basic()

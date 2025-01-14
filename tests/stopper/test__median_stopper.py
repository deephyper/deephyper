import numpy as np

from deephyper.evaluator import RunningJob
from deephyper.hpo import HpProblem
from deephyper.hpo import CBO, RandomSearch
from deephyper.stopper import MedianStopper


def run(job: RunningJob) -> dict:
    assert isinstance(job.stopper, MedianStopper)

    max_budget = 50
    objective_i = 0

    for budget_i in range(1, max_budget + 1):
        objective_i += job["x"]

        job.record(budget_i, objective_i)
        if job.stopped():
            break

    return {
        "objective": job.objective,
        "metadata": {"budget": budget_i, "stopped": budget_i < max_budget},
    }


def test_median_stopper_with_cbo(tmp_path):
    # define the variable you want to optimize
    problem = HpProblem()
    problem.add_hyperparameter((0.0, 10.0), "x")

    stopper = MedianStopper(max_steps=50, min_steps=1)
    search = CBO(
        problem,
        run,
        surrogate_model="DUMMY",
        stopper=stopper,
        random_state=42,
        log_dir=tmp_path,
    )

    results = search.search(max_evals=30)

    assert "m:budget" in results.columns
    assert "m:stopped" in results.columns
    assert "p:x" in results.columns
    assert "objective" in results.columns

    budgets = np.sort(np.unique(results["m:budget"].to_numpy())).tolist()
    assert max(budgets) == 50
    assert len(budgets) > 1
    assert results["m:budget"].sum() < 50 * 30


def test_median_stopper_with_random_search(tmp_path):
    # define the variable you want to optimize
    problem = HpProblem()
    problem.add_hyperparameter((0.0, 10.0), "x")

    stopper = MedianStopper(max_steps=50, min_steps=1)
    search = RandomSearch(
        problem,
        run,
        random_state=42,
        log_dir=tmp_path,
        stopper=stopper,
    )

    results = search.search(max_evals=30)

    assert "m:budget" in results.columns
    assert "m:stopped" in results.columns
    assert "p:x" in results.columns
    assert "objective" in results.columns

    budgets = np.sort(np.unique(results["m:budget"].to_numpy())).tolist()
    assert max(budgets) == 50
    assert len(budgets) > 1
    assert results["m:budget"].sum() < 50 * 30


if __name__ == "__main__":
    # test_median_stopper_with_cbo(tmp_path="/tmp/deephyper_test")
    test_median_stopper_with_random_search(tmp_path="/tmp/deephyper_test")

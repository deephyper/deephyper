import pytest
import time

import numpy as np
import pandas as pd

from deephyper.evaluator import RunningJob
from deephyper.hpo import HpProblem
from deephyper.hpo import CBO
from deephyper.stopper import SuccessiveHalvingStopper


def run(job: RunningJob) -> dict:
    assert isinstance(job.stopper, SuccessiveHalvingStopper)

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


def test_successive_halving_stopper(tmp_path):
    # define the variable you want to optimize
    problem = HpProblem()
    problem.add_hyperparameter((0.0, 10.0), "x")

    stopper = SuccessiveHalvingStopper(max_steps=50, reduction_factor=3)
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
    assert budgets == [1, 3, 9, 50]


def run_slow(job: RunningJob) -> dict:
    assert isinstance(job.stopper, SuccessiveHalvingStopper)

    max_budget = 50
    objective_i = 0

    for budget_i in range(1, max_budget + 1):
        objective_i += job["x"]

        time.sleep(0.001)

        job.record(budget_i, objective_i)
        if job.stopped():
            break

    # print(f"job {job.id} stopped at budget {budget_i} with objective {objective_i}")
    return {
        "objective": job.objective,
        "metadata": {"budget": budget_i, "stopped": budget_i < max_budget},
    }


@pytest.mark.slow
@pytest.mark.ray
def test_successive_halving_stopper_with_ray(tmp_path):
    import os
    import ray
    from deephyper.evaluator import Evaluator

    if ray.is_initialized():
        ray.shutdown()

    evaluator = Evaluator.create(
        run_slow,
        method="ray",
        method_kwargs={
            "num_cpus": 4,
            "num_cpus_per_task": 1,
            "ray_kwargs": {
                "runtime_env": {"working_dir": os.path.dirname(os.path.abspath(__file__))}
            },
        },
    )

    assert evaluator.num_workers == 4

    # define the variable you want to optimize
    problem = HpProblem()
    problem.add_hyperparameter((0.0, 10.0), "x")

    stopper = SuccessiveHalvingStopper(max_steps=50, reduction_factor=3)
    search = CBO(
        problem,
        evaluator,
        surrogate_model="RF",
        stopper=stopper,
        random_state=42,
        log_dir=tmp_path,
    )

    results = search.search(timeout=5)

    print(results)

    assert "m:budget" in results.columns
    assert "m:stopped" in results.columns
    assert "p:x" in results.columns
    assert "objective" in results.columns

    budgets = np.sort(np.unique(results["m:budget"].to_numpy())).tolist()
    assert any(b in budgets for b in [1, 3, 9, 27, 50])


def run_with_failures(job: RunningJob) -> dict:
    assert isinstance(job.stopper, SuccessiveHalvingStopper)

    max_budget = 50
    objective_i = 0
    for budget_i in range(1, max_budget + 1):
        objective_i += job["x"]

        if objective_i >= 450:
            objective_i = "F"

        job.record(budget_i, objective_i)
        if job.stopped():
            break

    return {
        "objective": job.objective,
        "metadata": {"budget": budget_i, "stopped": budget_i < max_budget},
    }


def test_successive_halving_stopper_with_failing_evaluations(tmp_path):
    # define the variable you want to optimize
    problem = HpProblem()
    problem.add_hyperparameter((0.0, 10.0), "x")

    stopper = SuccessiveHalvingStopper(max_steps=50, reduction_factor=3)
    search = CBO(
        problem,
        run_with_failures,
        surrogate_model="RF",
        stopper=stopper,
        random_state=42,
        filter_failures="mean",
        log_dir=tmp_path,
    )

    results = search.search(max_evals=50)

    assert "m:budget" in results.columns
    assert "m:stopped" in results.columns
    assert "p:x" in results.columns
    assert "objective" in results.columns

    assert pd.api.types.is_string_dtype(results.objective)

    results = results[~results.objective.str.startswith("F")]
    results.objective = results.objective.astype(float)

    # The constraint inside the run-function should make the job fail if > 450
    assert results.objective.max() < 450

    # Test the optimization worked
    assert results.objective.max() > 449


if __name__ == "__main__":
    # test_successive_halving_stopper(tmp_path=".")
    test_successive_halving_stopper_with_ray(tmp_path=".")
    # test_successive_halving_stopper_with_failing_evaluations(tmp_path=".")

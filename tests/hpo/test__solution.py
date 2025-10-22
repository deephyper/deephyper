import numpy as np
import pytest

from deephyper.hpo import CBO, ArgMaxEstSelection, ArgMaxObsSelection, HpProblem

CBO_DEFAULT_KWARGS = dict(
    random_state=42,
    surrogate_model="ET",
    surrogate_model_kwargs={"n_estimators": 25, "min_samples_leaf": 3},
    acq_func_kwargs={"kappa": 1.96},
    acq_optimizer="sampling",
    acq_optimizer_kwargs={"n_points": 100},
)


# Monkey patch grid search
def get_parameter_grid(self) -> dict:
    # Default grid for ExtraTrees
    p_grid = {
        "splitter": ["random"],
        "n_estimators": [25],
        "bootstrap": [True],
        "min_samples_leaf": [4],
        "min_samples_split": [8],
        "max_depth": [None],
    }
    return p_grid


ArgMaxEstSelection.get_parameter_grid = get_parameter_grid


def run_simple(job):
    return job.parameters["x"]


def run_with_failures(job):
    if job.parameters["x"] < 0.5:
        return "F_failed"

    return job.parameters["x"]


def run_multi_objective(job):
    y0 = job.parameters["x"]
    y1 = 1 - job.parameters["x"]
    return y0, y1


def test_cbo_solution_selection():
    problem = HpProblem()
    problem.add_hyperparameter((0.0, 1.0), "x")

    max_evals = 25

    # Test 1: with default value for "solution_selection"
    search = CBO(problem, checkpoint_history_to_csv=False, **CBO_DEFAULT_KWARGS)
    results = search.search(run_simple, max_evals)

    assert "sol.p:x" not in results.columns
    assert "sol.objective" not in results.columns

    # Test 2: with default value solution_selection="argmax_obs"
    search = CBO(
        problem,
        checkpoint_history_to_csv=False,
        solution_selection="argmax_obs",
        **CBO_DEFAULT_KWARGS,
    )
    results = search.search(run_simple, max_evals)

    assert "sol.p:x" in results.columns
    assert "sol.objective" in results.columns
    idx = results.objective.argmax()
    assert results["sol.objective"].iloc[-1] == results.objective.iloc[idx]
    assert np.all(results["sol.p:x"] == results["sol.objective"])

    # Test 3: wrong parameter value
    with pytest.raises(ValueError):
        search = CBO(
            problem,
            checkpoint_history_to_csv=False,
            solution_selection="helloworld",
            **CBO_DEFAULT_KWARGS,
        )

    # test 4: solution_selection="argmax_est"
    search = CBO(
        problem,
        checkpoint_history_to_csv=False,
        solution_selection="argmax_est",
        **CBO_DEFAULT_KWARGS,
    )
    results = search.search(run_simple, max_evals)

    assert "sol.p:x" in results.columns
    assert "sol.objective" in results.columns
    assert np.any(results["sol.p:x"] != results["sol.objective"])

    # test 5: with instance of ArgMaxObsSelection
    search = CBO(
        problem,
        checkpoint_history_to_csv=False,
        solution_selection=ArgMaxObsSelection(),
        **CBO_DEFAULT_KWARGS,
    )
    results = search.search(run_simple, max_evals)

    assert "sol.p:x" in results.columns
    assert "sol.objective" in results.columns
    idx = results.objective.argmax()
    assert results["sol.objective"].iloc[-1] == results.objective.iloc[idx]
    assert np.all(results["sol.p:x"] == results["sol.objective"])

    # test 6: with instance of ArgMaxEstSelection
    search = CBO(
        problem,
        checkpoint_history_to_csv=False,
        solution_selection=ArgMaxEstSelection(problem, 42, "RF", {"n_estimators": 25}),
        **CBO_DEFAULT_KWARGS,
    )
    results = search.search(run_simple, max_evals)
    assert "sol.p:x" in results.columns
    assert "sol.objective" in results.columns
    assert np.any(results["sol.p:x"] != results["sol.objective"])

    # test 7: with instance of ArgMaxEstSelection and sampling optimization
    search = CBO(
        problem,
        checkpoint_history_to_csv=False,
        solution_selection=ArgMaxEstSelection(
            problem,
            42,
            "RF",
            {"n_estimators": 25},
            optimizer="sampling",
        ),
        **CBO_DEFAULT_KWARGS,
    )
    results = search.search(run_simple, max_evals)
    assert "sol.p:x" in results.columns
    assert "sol.objective" in results.columns
    assert np.any(results["sol.p:x"] != results["sol.objective"])

    # test 8: with instance of ArgMaxEstSelection grid search and noisy
    solution_selection = ArgMaxEstSelection(
        problem,
        42,
        "RF",
        {"n_estimators": 25},
        model_grid_search_period=10,
        noisy_objective=True,
    )
    search = CBO(
        problem,
        checkpoint_history_to_csv=False,
        solution_selection=solution_selection,
        **CBO_DEFAULT_KWARGS,
    )
    results = search.search(run_simple, max_evals)
    assert "sol.p:x" in results.columns
    assert "sol.objective" in results.columns
    assert np.any(results["sol.p:x"] != results["sol.objective"])


def test_cbo_solution_selection_with_failures():
    problem = HpProblem()
    problem.add_hyperparameter((0.0, 1.0), "x")

    max_evals = 25

    # Test for "argmax_obs"
    search = CBO(
        problem,
        checkpoint_history_to_csv=False,
        solution_selection="argmax_obs",
        **CBO_DEFAULT_KWARGS,
    )
    results = search.search(run_with_failures, max_evals)

    assert "sol.p:x" in results.columns
    assert "sol.objective" in results.columns

    # Test for "argmax_est"
    search = CBO(
        problem,
        checkpoint_history_to_csv=False,
        solution_selection="argmax_est",
        **CBO_DEFAULT_KWARGS,
    )
    results = search.search(run_with_failures, max_evals)

    assert "sol.p:x" in results.columns
    assert "sol.objective" in results.columns


def test_cbo_solution_selection_with_moo():
    problem = HpProblem()
    problem.add_hyperparameter((0.0, 1.0), "x")

    max_evals = 25

    # test 1: with default value for "solution_selection"
    search = CBO(
        problem,
        checkpoint_history_to_csv=False,
        solution_selection="argmax_est",
        **CBO_DEFAULT_KWARGS,
    )
    results = search.search(run_multi_objective, max_evals)

    # The solution selection for multi-objective is not implemented yet
    # Therefore the columns "sol." should not appear for now
    assert "sol.p:x" not in results.columns
    assert "sol.objective" not in results.columns


if __name__ == "__main__":
    test_cbo_solution_selection()
    # test_cbo_solution_selection_with_failures()
    # test_cbo_solution_selection_with_moo()

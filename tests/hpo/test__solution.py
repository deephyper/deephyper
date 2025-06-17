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


def run(job):
    return job.parameters["x"]


def test_cbo_solution_selection():
    problem = HpProblem()
    problem.add_hyperparameter((0.0, 1.0), "x")

    max_evals = 25

    # test 1: with default value for "solution_selection"
    search = CBO(problem, run, checkpoint_history_to_csv=False, **CBO_DEFAULT_KWARGS)
    results = search.search(max_evals)

    assert "sol.p:x" in results.columns
    assert "sol.objective" in results.columns
    idx = results.objective.argmax()
    assert results["sol.objective"].iloc[-1] == results.objective.iloc[idx]
    assert np.all(results["sol.p:x"] == results["sol.objective"])

    # test 2: with default value solution_selection="argmax_obs"
    search = CBO(
        problem,
        run,
        checkpoint_history_to_csv=False,
        solution_selection="argmax_obs",
        **CBO_DEFAULT_KWARGS,
    )
    results = search.search(max_evals)

    assert "sol.p:x" in results.columns
    assert "sol.objective" in results.columns
    idx = results.objective.argmax()
    assert results["sol.objective"].iloc[-1] == results.objective.iloc[idx]
    assert np.all(results["sol.p:x"] == results["sol.objective"])

    # test 3: wrong parameter value
    with pytest.raises(ValueError):
        search = CBO(
            problem,
            run,
            checkpoint_history_to_csv=False,
            solution_selection="helloworld",
            **CBO_DEFAULT_KWARGS,
        )

    # test 4: solution_selection="argmax_est"
    search = CBO(
        problem,
        run,
        checkpoint_history_to_csv=False,
        solution_selection="argmax_est",
        **CBO_DEFAULT_KWARGS,
    )
    results = search.search(max_evals)

    assert "sol.p:x" in results.columns
    assert "sol.objective" in results.columns
    assert np.any(results["sol.p:x"] != results["sol.objective"])

    # test 5: with instance of ArgMaxObsSelection
    search = CBO(
        problem,
        run,
        checkpoint_history_to_csv=False,
        solution_selection=ArgMaxObsSelection(),
        **CBO_DEFAULT_KWARGS,
    )
    results = search.search(max_evals)

    assert "sol.p:x" in results.columns
    assert "sol.objective" in results.columns
    idx = results.objective.argmax()
    assert results["sol.objective"].iloc[-1] == results.objective.iloc[idx]
    assert np.all(results["sol.p:x"] == results["sol.objective"])

    # test 6: with instance of ArgMaxEstSelection
    search = CBO(
        problem,
        run,
        checkpoint_history_to_csv=False,
        solution_selection=ArgMaxEstSelection(problem, 42, "RF", {"n_estimators": 25}),
        **CBO_DEFAULT_KWARGS,
    )
    results = search.search(max_evals)
    assert "sol.p:x" in results.columns
    assert "sol.objective" in results.columns
    assert np.any(results["sol.p:x"] != results["sol.objective"])


if __name__ == "__main__":
    test_cbo_solution_selection()

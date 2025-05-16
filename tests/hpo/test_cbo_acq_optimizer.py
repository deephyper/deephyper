import numpy as np
import pytest
from ConfigSpace import (
    ForbiddenAndConjunction,
    ForbiddenEqualsClause,
    ForbiddenEqualsRelation,
    GreaterThanCondition,
)

from deephyper.hpo import CBO, HpProblem

SEARCH_KWARGS_DEFAULTS = dict(
    n_points=100,
    random_state=42,
    surrogate_model="ET",
    surrogate_model_kwargs={"n_estimators": 25, "min_samples_split": 8},
)


def test_pymoo_ga(tmp_path):
    problem = HpProblem()
    problem.add_hyperparameter((10.0, 20.0), "x")

    def run(job):
        return job.parameters["x"]

    search = CBO(
        problem,
        run,
        verbose=0,
        log_dir=tmp_path,
        random_state=SEARCH_KWARGS_DEFAULTS["random_state"],
        surrogate_model=SEARCH_KWARGS_DEFAULTS["surrogate_model"],
        surrogate_model_kwargs=SEARCH_KWARGS_DEFAULTS["surrogate_model_kwargs"],
        acq_optimizer="ga",
        acq_optimizer_kwargs=dict(
            acq_optimizer_freq=1,
            n_points=SEARCH_KWARGS_DEFAULTS["n_points"],
        ),
        acq_func_kwargs=dict(
            kappa=5.0, scheduler={"type": "periodic-exp-decay", "period": 25, "kappa_final": 0.0001}
        ),
    )
    results = search.search(max_evals=25)

    assert len(results) == 25
    assert len(results["objective"] > 19) > 8


def test_pymoo_mixedga(tmp_path):
    problem = HpProblem()
    problem.add_hyperparameter((10.0, 20.0), "x_real")
    problem.add_hyperparameter((10, 20), "x_int")
    problem.add_hyperparameter([f"{i}" for i in range(10)], "x_cat")

    def run(job):
        return job.parameters["x_real"] + job.parameters["x_int"] + int(job.parameters["x_cat"])

    search = CBO(
        problem,
        run,
        log_dir=tmp_path,
        verbose=0,
        random_state=SEARCH_KWARGS_DEFAULTS["random_state"],
        surrogate_model=SEARCH_KWARGS_DEFAULTS["surrogate_model"],
        surrogate_model_kwargs=SEARCH_KWARGS_DEFAULTS["surrogate_model_kwargs"],
        acq_optimizer="mixedga",
        acq_optimizer_kwargs=dict(
            acq_optimizer_freq=1,
            n_points=SEARCH_KWARGS_DEFAULTS["n_points"],
        ),
        acq_func_kwargs=dict(
            kappa=5.0, scheduler={"type": "periodic-exp-decay", "period": 25, "kappa_final": 0.0001}
        ),
    )
    results = search.search(max_evals=25)

    assert len(results) == 25
    assert len(results["objective"] > 40) > 8


@pytest.mark.slow
def test_cbo_with_acq_optimizer_mixedga_and_conditions_in_problem(tmp_path):
    problem = HpProblem()

    max_num_layers = 3
    num_layers = problem.add_hyperparameter((1, max_num_layers), "num_layers", default_value=2)

    conditions = []
    for i in range(max_num_layers):
        layer_i_units = problem.add_hyperparameter((1, 50), f"layer_{i}_units", default_value=32)

        if i > 0:
            conditions.extend(
                [
                    GreaterThanCondition(layer_i_units, num_layers, i),
                ]
            )
    problem.add_conditions(conditions)

    def run(job):
        num_layers = job.parameters["num_layers"]
        return sum(job.parameters[f"layer_{i}_units"] for i in range(num_layers))

    search = CBO(
        problem,
        run,
        n_points=SEARCH_KWARGS_DEFAULTS["n_points"],
        random_state=SEARCH_KWARGS_DEFAULTS["random_state"],
        surrogate_model=SEARCH_KWARGS_DEFAULTS["surrogate_model"],
        surrogate_model_kwargs=SEARCH_KWARGS_DEFAULTS["surrogate_model_kwargs"],
        log_dir=tmp_path,
        acq_optimizer="mixedga",
        acq_optimizer_freq=1,
        kappa=5.0,
        scheduler={"type": "periodic-exp-decay", "period": 25, "kappa_final": 0.0001},
        verbose=0,
    )
    results = search.search(max_evals=25)

    assert (results[(results["p:num_layers"] == 1)]["p:layer_1_units"] == 1).all()
    assert (results[(results["p:num_layers"] == 1)]["p:layer_2_units"] == 1).all()
    assert (results[(results["p:num_layers"] == 2)]["p:layer_2_units"] == 1).all()
    assert results["objective"].max() > 100


@pytest.mark.slow
def test_cbo_with_acq_optimizer_mixedga_and_forbiddens_in_problem(tmp_path):
    problem = HpProblem()

    max_num_layers = 5
    for i in range(max_num_layers):
        problem.add_hyperparameter(
            (0, max_num_layers), f"layer_{i}_units", default_value=(i + 1) % max_num_layers
        )
    forbiddens = []
    for i in range(1, max_num_layers):
        forb = ForbiddenEqualsRelation(problem[f"layer_{i - 1}_units"], problem[f"layer_{i}_units"])
        forbiddens.append(forb)
    problem.add_forbidden_clause(forbiddens)
    problem.add_forbidden_clause(
        ForbiddenAndConjunction(
            *(ForbiddenEqualsClause(problem[f"layer_{i}_units"], i) for i in range(max_num_layers))
        )
    )
    print(problem)

    def run(job):
        return sum(job.parameters[f"layer_{i}_units"] for i in range(max_num_layers))

    search = CBO(
        problem,
        run,
        n_points=SEARCH_KWARGS_DEFAULTS["n_points"],
        random_state=SEARCH_KWARGS_DEFAULTS["random_state"],
        surrogate_model=SEARCH_KWARGS_DEFAULTS["surrogate_model"],
        surrogate_model_kwargs=SEARCH_KWARGS_DEFAULTS["surrogate_model_kwargs"],
        log_dir=tmp_path,
        acq_optimizer="mixedga",
        acq_optimizer_freq=1,
        kappa=5.0,
        scheduler={"type": "periodic-exp-decay", "period": 25, "kappa_final": 0.0001},
        verbose=0,
    )
    results = search.search(max_evals=25)

    cond = np.ones(len(results), dtype=bool)
    for i in range(1, max_num_layers):
        assert (results[f"p:layer_{i - 1}_units"] != results[f"p:layer_{i}_units"]).all(), (
            f"for {i=}"
        )
        cond = cond & (results[f"p:layer_{i - 1}_units"] == i)
    assert (~cond).all()


if __name__ == "__main__":
    test_pymoo_mixedga(".")

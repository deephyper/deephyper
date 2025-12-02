import os
import time

import ConfigSpace as cs
import numpy as np
import pytest

from deephyper.evaluator import Evaluator
from deephyper.hpo import CBO, HpProblem

SEARCH_KWARGS_DEFAULTS = dict(
    random_state=42,
    surrogate_model="ET",
    surrogate_model_kwargs={"n_estimators": 25, "min_samples_split": 8},
    acq_func_kwargs={"kappa": "toto"},
    acq_optimizer="sampling",
    acq_optimizer_kwargs={"n_points": 100},
)


def test_cbo_random_seed(tmp_path):
    problem = HpProblem()
    problem.add_hyperparameter((0.0, 10.0), "x")

    async def run_single_objective(config):
        return config["x"]

    def create_evaluator():
        return Evaluator.create(run_single_objective, method="serial")

    search = CBO(
        problem,
        acq_optimizer=SEARCH_KWARGS_DEFAULTS["acq_optimizer"],
        acq_optimizer_kwargs=SEARCH_KWARGS_DEFAULTS["acq_optimizer_kwargs"],
        random_state=SEARCH_KWARGS_DEFAULTS["random_state"],
        surrogate_model="DUMMY",
        log_dir=tmp_path,
    )

    res1 = search.search(create_evaluator(), max_evals=4)
    res1_array = res1[["p:x"]].to_numpy()

    search = CBO(
        problem,
        acq_optimizer=SEARCH_KWARGS_DEFAULTS["acq_optimizer"],
        acq_optimizer_kwargs=SEARCH_KWARGS_DEFAULTS["acq_optimizer_kwargs"],
        random_state=SEARCH_KWARGS_DEFAULTS["random_state"],
        surrogate_model="DUMMY",
        log_dir=tmp_path,
    )
    res2 = search.search(create_evaluator(), max_evals=4)
    res2_array = res2[["p:x"]].to_numpy()

    assert np.array_equal(res1_array, res2_array)

    # test multi-objective
    async def run_multi_objective(config):
        return config["x"], config["x"]

    def create_evaluator():
        return Evaluator.create(run_multi_objective, method="serial")

    search = CBO(
        problem,
        acq_optimizer=SEARCH_KWARGS_DEFAULTS["acq_optimizer"],
        acq_optimizer_kwargs=SEARCH_KWARGS_DEFAULTS["acq_optimizer_kwargs"],
        random_state=SEARCH_KWARGS_DEFAULTS["random_state"],
        surrogate_model="DUMMY",
        log_dir=tmp_path,
    )

    res1 = search.search(create_evaluator(), max_evals=4)
    res1_array = res1[["p:x"]].to_numpy()

    search = CBO(
        problem,
        acq_optimizer=SEARCH_KWARGS_DEFAULTS["acq_optimizer"],
        acq_optimizer_kwargs=SEARCH_KWARGS_DEFAULTS["acq_optimizer_kwargs"],
        random_state=SEARCH_KWARGS_DEFAULTS["random_state"],
        surrogate_model="DUMMY",
        log_dir=tmp_path,
    )
    res2 = search.search(create_evaluator(), max_evals=4)
    res2_array = res2[["p:x"]].to_numpy()

    assert np.array_equal(res1_array, res2_array)


def test_sample_types(tmp_path):
    problem = HpProblem()
    problem.add_hyperparameter((0, 10), "x_int")
    problem.add_hyperparameter((0.0, 10.0), "x_float")
    problem.add_hyperparameter(["0", "1", "2"], "x_cat")

    def run(config):
        assert np.issubdtype(type(config["x_int"]), np.integer)
        assert np.issubdtype(type(config["x_float"]), float)
        assert type(config["x_cat"]) is str or type(config["x_cat"]) is np.str_

        return config["x_int"] + config["x_float"] + int(config["x_cat"])

    max_evals = 20
    results = CBO(
        problem,
        n_initial_points=5,
        acq_optimizer=SEARCH_KWARGS_DEFAULTS["acq_optimizer"],
        acq_optimizer_kwargs=SEARCH_KWARGS_DEFAULTS["acq_optimizer_kwargs"],
        random_state=SEARCH_KWARGS_DEFAULTS["random_state"],
        surrogate_model="DUMMY",
        log_dir=tmp_path,
        verbose=0,
    ).search(run, max_evals)
    assert len(results) == max_evals

    results = CBO(
        problem,
        n_initial_points=5,
        acq_optimizer=SEARCH_KWARGS_DEFAULTS["acq_optimizer"],
        acq_optimizer_kwargs=SEARCH_KWARGS_DEFAULTS["acq_optimizer_kwargs"],
        random_state=SEARCH_KWARGS_DEFAULTS["random_state"],
        surrogate_model=SEARCH_KWARGS_DEFAULTS["surrogate_model"],
        surrogate_model_kwargs=SEARCH_KWARGS_DEFAULTS["surrogate_model_kwargs"],
        log_dir=tmp_path,
        verbose=0,
    ).search(run, max_evals)
    assert len(results) == max_evals


def test_sample_types_no_cat(tmp_path):
    problem = HpProblem()
    problem.add_hyperparameter((0, 10), "x_int")
    problem.add_hyperparameter((0.0, 10.0), "x_float")

    def run(config):
        assert np.issubdtype(type(config["x_int"]), np.integer)
        assert np.issubdtype(type(config["x_float"]), float)

        return config["x_int"] + config["x_float"]

    max_evals = 20
    results = CBO(
        problem,
        random_state=SEARCH_KWARGS_DEFAULTS["random_state"],
        acq_optimizer=SEARCH_KWARGS_DEFAULTS["acq_optimizer"],
        acq_optimizer_kwargs=SEARCH_KWARGS_DEFAULTS["acq_optimizer_kwargs"],
        n_initial_points=5,
        surrogate_model="DUMMY",
        log_dir=tmp_path,
        verbose=0,
    ).search(run, max_evals)
    assert len(results) == max_evals

    results = CBO(
        problem,
        random_state=SEARCH_KWARGS_DEFAULTS["random_state"],
        acq_optimizer=SEARCH_KWARGS_DEFAULTS["acq_optimizer"],
        acq_optimizer_kwargs=SEARCH_KWARGS_DEFAULTS["acq_optimizer_kwargs"],
        n_initial_points=5,
        surrogate_model=SEARCH_KWARGS_DEFAULTS["surrogate_model"],
        surrogate_model_kwargs=SEARCH_KWARGS_DEFAULTS["surrogate_model_kwargs"],
        log_dir=tmp_path,
        verbose=0,
    ).search(run, max_evals)
    assert len(results) == max_evals


def test_gp(tmp_path):
    # test float hyperparameters
    problem = HpProblem()
    problem.add_hyperparameter((0.0, 10.0), "x")

    async def run(config):
        return config["x"]

    max_evals = 20
    results = CBO(
        problem,
        acq_optimizer="lbfgs",
        acq_optimizer_kwargs=SEARCH_KWARGS_DEFAULTS["acq_optimizer_kwargs"],
        random_state=SEARCH_KWARGS_DEFAULTS["random_state"],
        surrogate_model="GP",
        acq_func="UCB",
        log_dir=tmp_path,
    ).search(Evaluator.create(run, method="serial"), max_evals)
    assert len(results) == max_evals

    # test int hyperparameters
    problem = HpProblem()
    problem.add_hyperparameter((0, 10), "x")

    results = CBO(
        problem,
        acq_optimizer="lbfgs",
        acq_optimizer_kwargs=SEARCH_KWARGS_DEFAULTS["acq_optimizer_kwargs"],
        random_state=SEARCH_KWARGS_DEFAULTS["random_state"],
        surrogate_model="GP",
        acq_func="UCB",
        log_dir=tmp_path,
    ).search(Evaluator.create(run, method="serial"), max_evals)
    assert len(results) == max_evals

    # test categorical hyperparameters
    problem = HpProblem()
    problem.add_hyperparameter([f"{i}" for i in range(10)], "x")

    async def run_cast_output_int(config):
        return int(config["x"])

    results = CBO(
        problem,
        acq_optimizer="lbfgs",
        acq_optimizer_kwargs=SEARCH_KWARGS_DEFAULTS["acq_optimizer_kwargs"],
        random_state=SEARCH_KWARGS_DEFAULTS["random_state"],
        surrogate_model="GP",
        acq_func="UCB",
        log_dir=tmp_path,
    ).search(Evaluator.create(run_cast_output_int, method="serial"), max_evals)
    assert len(results) == max_evals


def test_sample_types_conditional(tmp_path):
    problem = HpProblem()

    # choices
    choice = problem.add_hyperparameter(
        name="choice",
        value=["choice1", "choice2", "choice3"],
    )

    # integers
    x1_int = problem.add_hyperparameter(name="x1_int", value=(1, 10))

    x2_float = problem.add_hyperparameter(name="x2_float", value=(1.0, 10.0))

    x3_cat = problem.add_hyperparameter(name="x3_cat", value=["1_", "2_", "3_"])

    # conditions
    cond_1 = cs.EqualsCondition(x1_int, choice, "choice1")
    cond_2 = cs.EqualsCondition(x2_float, choice, "choice2")
    cond_3 = cs.EqualsCondition(x3_cat, choice, "choice3")
    problem.add_conditions([cond_1, cond_2, cond_3])

    def run(config):
        if config["choice"] == "choice1":
            assert np.issubdtype(type(config["x1_int"]), np.integer)
            y = config["x1_int"] ** 2
        elif config["choice"] == "choice2":
            assert np.issubdtype(type(config["x2_float"]), float)
            y = config["x2_float"] ** 2 + 1
        else:
            assert type(config["x3_cat"]) is str or type(config["x3_cat"]) is np.str_
            y = int(config["x3_cat"][:1]) ** 2 + 2

        return y

    # Test classic random search
    search = CBO(
        problem,
        acq_optimizer=SEARCH_KWARGS_DEFAULTS["acq_optimizer"],
        acq_optimizer_kwargs=SEARCH_KWARGS_DEFAULTS["acq_optimizer_kwargs"],
        random_state=SEARCH_KWARGS_DEFAULTS["random_state"],
        surrogate_model="DUMMY",
        log_dir=tmp_path,
        verbose=0,
    )
    results = search.search(run, 10)
    assert len(results) == 10

    # Test search with ET surrogate model
    results = CBO(
        problem,
        acq_optimizer=SEARCH_KWARGS_DEFAULTS["acq_optimizer"],
        acq_optimizer_kwargs=SEARCH_KWARGS_DEFAULTS["acq_optimizer_kwargs"],
        random_state=SEARCH_KWARGS_DEFAULTS["random_state"],
        n_initial_points=5,
        surrogate_model=SEARCH_KWARGS_DEFAULTS["surrogate_model"],
        surrogate_model_kwargs=SEARCH_KWARGS_DEFAULTS["surrogate_model_kwargs"],
        log_dir=tmp_path,
        verbose=1,
    ).search(run, 20)
    assert len(results) == 20


def run_max_evals(job):
    config = job.parameters
    return config["x"]


def test_max_evals_strict(tmp_path):
    problem = HpProblem()
    problem.add_hyperparameter((0.0, 10.0), "x")

    evaluator = Evaluator.create(run_max_evals, method="process", method_kwargs={"num_workers": 8})

    # Test Timeout with max_evals (this should be like an "max_evals or timeout" condition)
    search = CBO(
        problem,
        acq_optimizer=SEARCH_KWARGS_DEFAULTS["acq_optimizer"],
        acq_optimizer_kwargs=SEARCH_KWARGS_DEFAULTS["acq_optimizer_kwargs"],
        random_state=SEARCH_KWARGS_DEFAULTS["random_state"],
        surrogate_model="DUMMY",
        log_dir=tmp_path,
    )

    max_evals = 100
    results = search.search(evaluator, max_evals=max_evals, max_evals_strict=True)
    assert len(results) == max_evals


def test_initial_points(tmp_path):
    problem = HpProblem()
    problem.add_hyperparameter((0.0, 10.0), "x")

    def run(config):
        return config["x"]

    search = CBO(
        problem,
        initial_points=[problem.default_configuration],
        acq_optimizer=SEARCH_KWARGS_DEFAULTS["acq_optimizer"],
        acq_optimizer_kwargs=SEARCH_KWARGS_DEFAULTS["acq_optimizer_kwargs"],
        random_state=SEARCH_KWARGS_DEFAULTS["random_state"],
        surrogate_model="DUMMY",
        log_dir=tmp_path,
    )

    result = search.search(run, 10)
    assert len(result) == 10
    assert result.loc[0, "objective"] == problem.default_configuration["x"]


def test_many_initial_points(tmp_path):
    problem = HpProblem()
    problem.add_hyperparameter((0.0, 10.0), "x")

    def run(config):
        return config["x"]

    max_evals = 100
    search = CBO(
        problem,
        initial_points=[{"x": v} for v in np.linspace(0.0, 10.0, max_evals)],
        acq_optimizer="sampling",
        random_state=SEARCH_KWARGS_DEFAULTS["random_state"],
        surrogate_model="DUMMY",
        log_dir=tmp_path,
    )

    result = search.search(run, max_evals=max_evals, max_evals_strict=True)
    assert len(result) == max_evals


def test_cbo_checkpoint_restart(tmp_path):
    problem = HpProblem()
    problem.add_hyperparameter((0, 10), "x_int")
    problem.add_hyperparameter((0.0, 10.0), "x_float")
    problem.add_hyperparameter(["a", "b", "c"], "x_cat")

    def run(job):
        objective = (
            job.parameters["x_int"] + job.parameters["x_float"] + ord(job.parameters["x_cat"])
        )
        return objective

    search_kwargs = dict(
        problem=problem,
        initial_points=[problem.default_configuration],
        random_state=SEARCH_KWARGS_DEFAULTS["random_state"],
        acq_optimizer=SEARCH_KWARGS_DEFAULTS["acq_optimizer"],
        acq_optimizer_kwargs=SEARCH_KWARGS_DEFAULTS["acq_optimizer_kwargs"],
    )

    def get_log_dir(name):
        return os.path.join(tmp_path, name)

    # test pause-continue of the search
    search_a = CBO(
        surrogate_model="DUMMY",
        log_dir=get_log_dir("search_a"),
        **search_kwargs,
    )

    results_a = search_a.search(run, 4)
    assert len(results_a) == 4

    new_results_a = search_a.search(run, 6)
    assert all(results_a["p:x_int"] == new_results_a.iloc[:4]["p:x_int"])
    assert len(new_results_a) == 10

    # test reloading of a checkpoint directly as dataframe
    search_b = CBO(
        surrogate_model=SEARCH_KWARGS_DEFAULTS["surrogate_model"],
        surrogate_model_kwargs=SEARCH_KWARGS_DEFAULTS["surrogate_model_kwargs"],
        log_dir=get_log_dir("search_b"),
        **search_kwargs,
    )

    search_b.fit_surrogate(results_a)
    new_results_b = search_b.search(run, 6)
    assert len(new_results_b) == 6

    # test reloading of a checkpoint from a file
    search_c = CBO(
        surrogate_model=SEARCH_KWARGS_DEFAULTS["surrogate_model"],
        surrogate_model_kwargs=SEARCH_KWARGS_DEFAULTS["surrogate_model_kwargs"],
        log_dir=get_log_dir("search_c"),
        **search_kwargs,
    )
    search_c.fit_surrogate(os.path.join(get_log_dir("search_b"), "results.csv"))
    results_c = search_c.search(run, 20)

    assert len(results_c) == 20


def test_cbo_checkpoint_restart_moo(tmp_path):
    problem = HpProblem()
    problem.add_hyperparameter((0, 10), "x_int")
    problem.add_hyperparameter((0.0, 10.0), "x_float")
    problem.add_hyperparameter(["a", "b", "c"], "x_cat")

    def run(job):
        objective0 = (
            job.parameters["x_int"] + job.parameters["x_float"] + ord(job.parameters["x_cat"])
        )
        objective1 = (
            -job.parameters["x_int"] - job.parameters["x_float"] - ord(job.parameters["x_cat"])
        )
        return objective0, objective1

    search_kwargs = dict(
        problem=problem,
        initial_points=[problem.default_configuration],
        random_state=SEARCH_KWARGS_DEFAULTS["random_state"],
        acq_optimizer_kwargs=SEARCH_KWARGS_DEFAULTS["acq_optimizer_kwargs"],
    )

    def get_log_dir(name):
        return os.path.join(tmp_path, name)

    # test pause-continue of the search
    search_a = CBO(
        log_dir=get_log_dir("search_a"),
        surrogate_model="DUMMY",
        **search_kwargs,
    )

    results_a = search_a.search(run, 4)
    assert len(results_a) == 4

    # a column must be named "pareto_efficient"
    assert "pareto_efficient" in results_a.columns

    # at lest one element must be set to True
    assert any(results_a["pareto_efficient"])

    new_results_a = search_a.search(run, 6)
    assert all(results_a["p:x_int"] == new_results_a.iloc[:4]["p:x_int"])
    assert len(new_results_a) == 10

    # test reloading of a checkpoint directly as dataframe
    search_b = CBO(
        log_dir=get_log_dir("search_b"),
        surrogate_model=SEARCH_KWARGS_DEFAULTS["surrogate_model"],
        surrogate_model_kwargs=SEARCH_KWARGS_DEFAULTS["surrogate_model_kwargs"],
        acq_optimizer=SEARCH_KWARGS_DEFAULTS["acq_optimizer"],
        **search_kwargs,
    )

    search_b.fit_surrogate(results_a)
    new_results_b = search_b.search(run, 6)
    assert len(new_results_b) == 6

    # test reloading of a checkpoint from a file
    search_c = CBO(
        log_dir=get_log_dir("search_c"),
        surrogate_model=SEARCH_KWARGS_DEFAULTS["surrogate_model"],
        surrogate_model_kwargs=SEARCH_KWARGS_DEFAULTS["surrogate_model_kwargs"],
        acq_optimizer=SEARCH_KWARGS_DEFAULTS["acq_optimizer"],
        **search_kwargs,
    )
    search_c.fit_surrogate(os.path.join(get_log_dir("search_b"), "results.csv"))
    results_c = search_c.search(run, 20)

    assert len(results_c) == 20


def test_cbo_checkpoint_restart_with_failures(tmp_path):
    problem = HpProblem()
    problem.add_hyperparameter((0, 10), "x_int")
    problem.add_hyperparameter((0.0, 10.0), "x_float")
    problem.add_hyperparameter(["a", "b", "c"], "x_cat")

    def run(job):
        objective = (
            job.parameters["x_int"] + job.parameters["x_float"] + ord(job.parameters["x_cat"])
        )
        if job.parameters["x_cat"] == "c":
            objective = "F"
        return objective

    search_kwargs = dict(
        problem=problem,
        initial_points=[problem.default_configuration],
        random_state=SEARCH_KWARGS_DEFAULTS["random_state"],
        acq_optimizer=SEARCH_KWARGS_DEFAULTS["acq_optimizer"],
        acq_optimizer_kwargs=SEARCH_KWARGS_DEFAULTS["acq_optimizer_kwargs"],
    )

    def get_log_dir(name):
        return os.path.join(tmp_path, name)

    # test pause-continue of the search
    search_a = CBO(
        log_dir=get_log_dir("search_a"),
        surrogate_model="DUMMY",
        **search_kwargs,
    )

    results_a = search_a.search(run, 10)
    assert len(results_a) == 10

    new_results_a = search_a.search(run, 10)
    assert len(new_results_a) == 20

    # test reloading of a checkpoint directly as dataframe
    search_b = CBO(
        log_dir=get_log_dir("search_b"),
        surrogate_model=SEARCH_KWARGS_DEFAULTS["surrogate_model"],
        surrogate_model_kwargs=SEARCH_KWARGS_DEFAULTS["surrogate_model_kwargs"],
        **search_kwargs,
    )

    search_b.fit_surrogate(results_a)
    new_results_b = search_b.search(run, 20)
    assert len(new_results_b) == 20

    # test reloading of a checkpoint from a file
    search_c = CBO(
        log_dir=get_log_dir("search_c"),
        surrogate_model=SEARCH_KWARGS_DEFAULTS["surrogate_model"],
        surrogate_model_kwargs=SEARCH_KWARGS_DEFAULTS["surrogate_model_kwargs"],
        **search_kwargs,
    )
    search_c.fit_surrogate(os.path.join(tmp_path, "search_b", "results.csv"))
    results_c = search_c.search(run, 20)

    assert len(results_c) == 20

    # test reloading of a checkpoint from a file with failures only
    results_c["objective"] = "F"
    search_d = CBO(
        surrogate_model=SEARCH_KWARGS_DEFAULTS["surrogate_model"],
        surrogate_model_kwargs=SEARCH_KWARGS_DEFAULTS["surrogate_model_kwargs"],
        log_dir=get_log_dir("search_d"),
        **search_kwargs,
    )
    search_d.fit_surrogate(results_c)
    results_d = search_d.search(run, 20)

    assert len(results_d) == 20


def test_cbo_checkpoint_restart_moo_with_failures(tmp_path):
    problem = HpProblem()
    problem.add_hyperparameter((0, 10), "x_int")
    problem.add_hyperparameter((0.0, 10.0), "x_float")
    problem.add_hyperparameter(["a", "b", "c"], "x_cat")

    def run(job):
        objective0 = (
            job.parameters["x_int"] + job.parameters["x_float"] + ord(job.parameters["x_cat"])
        )
        objective1 = (
            -job.parameters["x_int"] - job.parameters["x_float"] - ord(job.parameters["x_cat"])
        )
        if job.parameters["x_cat"] == "c":
            objective0 = "F"
            objective1 = "F"
        return objective0, objective1

    search_kwargs = dict(
        problem=problem,
        initial_points=[problem.default_configuration],
        random_state=SEARCH_KWARGS_DEFAULTS["random_state"],
        surrogate_model="DUMMY",
        acq_optimizer_kwargs=SEARCH_KWARGS_DEFAULTS["acq_optimizer_kwargs"],
    )

    # test pause-continue of the search
    search_a = CBO(
        log_dir=os.path.join(tmp_path, "search_a"),
        **search_kwargs,
    )

    results_a = search_a.search(run, 10)
    assert len(results_a) == 10

    new_results_a = search_a.search(run, 10)
    assert len(new_results_a) == 20

    # test reloading of a checkpoint directly as dataframe
    search_b = CBO(
        log_dir=os.path.join(tmp_path, "search_b"),
        **search_kwargs,
    )

    search_b.fit_surrogate(results_a)
    new_results_b = search_b.search(run, 20)
    assert len(new_results_b) == 20

    # test reloading of a checkpoint from a file
    search_c = CBO(
        log_dir=os.path.join(tmp_path, "search_c"),
        **search_kwargs,
    )
    search_c.fit_surrogate(os.path.join(tmp_path, "search_b", "results.csv"))
    results_c = search_c.search(run, 20)

    assert len(results_c) == 20

    # test with cl mutlipoint
    eval = Evaluator.create(run, method="thread", method_kwargs={"num_workers": 2})
    results_d = search_c.search(eval, 20)

    # this search also includes the 20 iterations of previous .search(...)
    assert len(results_d) >= 40


def test_max_total_failures():
    from deephyper.skopt.optimizer import ExhaustedFailures

    # Case 1 - only failures
    def run_only_failures(job):
        return "F"

    problem = HpProblem()
    problem.add_hyperparameter((0, 10000), "x")

    search = CBO(
        problem,
        acq_optimizer_kwargs={"max_total_failures": 20},
        checkpoint_history_to_csv=False,
    )

    with pytest.raises(ExhaustedFailures):
        results = search.search(run_only_failures, 21)

    results = search.history.to_dataframe()
    assert len(results) == 20

    # Case 2 - failures then ok
    # 5 failures, 5 successes, 5 failures
    # max_total_failures allowed is 10 to generate n_initial_points=10
    # 10 failures are process only 5 successes, the n_initial_points was not reached
    # therefore raising error
    def run_some_success(job):
        return "F" if job.parameters["x"] == 0 else job.parameters["x"]

    problem = HpProblem()
    problem.add_hyperparameter((0, 10000), "x")

    search = CBO(
        problem,
        acq_optimizer_kwargs={"max_total_failures": 10},
        checkpoint_history_to_csv=False,
        n_initial_points=10,
        initial_points=[{"x": 0} for i in range(5)]
        + [{"x": 1} for i in range(5)]
        + [{"x": 0} for i in range(5)],
    )

    with pytest.raises(ExhaustedFailures):
        results = search.search(run_some_success, 21)

    results = search.history.to_dataframe()
    assert len(results) == 15

    # Case 3 - multi objective all failures
    def run_multi_only_failures(job):
        return "F", "F"

    problem = HpProblem()
    problem.add_hyperparameter((0, 10000), "x")
    problem.add_hyperparameter((0, 10000), "y")

    search = CBO(
        problem,
        acq_optimizer_kwargs={"max_total_failures": 20},
        checkpoint_history_to_csv=False,
    )

    with pytest.raises(ExhaustedFailures):
        results = search.search(run_multi_only_failures, 21)

    results = search.history.to_dataframe()
    assert len(results) == 20

    # Case 4 - multi objective some failures

    def run_multi_some_success(job):
        if (job.parameters["x"] == 0) and (job.parameters["y"] == 0):
            return "F", "F"
        else:
            return job.parameters["x"], job.parameters["y"]

    problem = HpProblem()
    problem.add_hyperparameter((0, 10000), "x")
    problem.add_hyperparameter((0, 10000), "y")

    search = CBO(
        problem,
        acq_optimizer_kwargs={"max_total_failures": 10},
        checkpoint_history_to_csv=False,
        n_initial_points=10,
        initial_points=[{"x": 0, "y": 0} for i in range(5)]
        + [{"x": 1, "y": 1} for i in range(5)]
        + [{"x": 0, "y": 0} for i in range(5)],
    )

    with pytest.raises(ExhaustedFailures):
        results = search.search(run_multi_some_success, 21)

    results = search.history.to_dataframe()
    assert len(results) == 15


def test_cbo_categorical_variable(tmp_path):
    from deephyper.evaluator import SerialEvaluator

    problem = HpProblem()
    problem.add_hyperparameter([32, 64, 96], "x", default_value=64)
    problem.add_hyperparameter((0.0, 10.0), "y", default_value=5.0)

    async def run(config):
        return config["x"] + config["y"]

    # test pause-continue of the search
    search = CBO(
        problem,
        initial_points=[problem.default_configuration],
        acq_optimizer=SEARCH_KWARGS_DEFAULTS["acq_optimizer"],
        acq_optimizer_kwargs=SEARCH_KWARGS_DEFAULTS["acq_optimizer_kwargs"],
        random_state=SEARCH_KWARGS_DEFAULTS["random_state"],
        surrogate_model=SEARCH_KWARGS_DEFAULTS["surrogate_model"],
        log_dir=tmp_path,
    )

    results = search.search(SerialEvaluator(run, callbacks=[]), 25)
    assert results.objective.max() >= 105


def test_cbo_multi_point_strategy(tmp_path):
    problem = HpProblem()
    problem.add_hyperparameter((0.0, 10.0), "x")

    async def run(config):
        return config["x"]

    durations = []
    for multi_point_strategy in ["cl_min", "cl_mean", "cl_max", "qUCB", "qUCBd"]:
        t1 = time.monotonic()
        search = CBO(
            problem,
            acq_optimizer=SEARCH_KWARGS_DEFAULTS["acq_optimizer"],
            acq_optimizer_kwargs=SEARCH_KWARGS_DEFAULTS["acq_optimizer_kwargs"],
            random_state=SEARCH_KWARGS_DEFAULTS["random_state"],
            surrogate_model=SEARCH_KWARGS_DEFAULTS["surrogate_model"],
            multi_point_strategy=multi_point_strategy,
            log_dir=tmp_path,
        )
        max_evals = 50
        results = search.search(
            Evaluator.create(run, method="serial", method_kwargs={"num_workers": 5}),
            max_evals,
            max_evals_strict=True,
        )
        durations.append(time.monotonic() - t1)

        assert len(results) == max_evals

    assert all(durations[i] > durations[j] for i in range(3) for j in range(3, 5))

    durations = []
    for multi_point_strategy in ["cl_min", "cl_mean", "cl_max", "qUCB", "qUCBd"]:
        t1 = time.monotonic()
        search = CBO(
            problem,
            acq_optimizer=SEARCH_KWARGS_DEFAULTS["acq_optimizer"],
            acq_optimizer_kwargs=SEARCH_KWARGS_DEFAULTS["acq_optimizer_kwargs"],
            acq_func="EI",
            random_state=SEARCH_KWARGS_DEFAULTS["random_state"],
            surrogate_model=SEARCH_KWARGS_DEFAULTS["surrogate_model"],
            surrogate_model_kwargs=SEARCH_KWARGS_DEFAULTS["surrogate_model_kwargs"],
            multi_point_strategy=multi_point_strategy,
            log_dir=tmp_path,
        )
        max_evals = 25
        results = search.search(
            Evaluator.create(run, method="serial", method_kwargs={"num_workers": 5}), 25
        )
        durations.append(time.monotonic() - t1)

        assert len(results) == 25

    assert all(durations[i] > durations[j] for i in range(3) for j in range(3, 5))


def test_cbo_fit_generative_model(tmp_path):
    problem = HpProblem()
    problem.add_hyperparameter((0, 10), "x_int")
    problem.add_hyperparameter((0.0, 10.0), "x_float")
    problem.add_hyperparameter(["a", "b", "c"], "x_cat")

    def run(job):
        return sum(ord(v) if isinstance(v, str) else v for v in job.parameters.values())

    search = CBO(
        problem,
        acq_optimizer_kwargs=SEARCH_KWARGS_DEFAULTS["acq_optimizer_kwargs"],
        random_state=SEARCH_KWARGS_DEFAULTS["random_state"],
        surrogate_model="DUMMY",
        log_dir=os.path.join(tmp_path, "search_0"),
        verbose=0,
    )
    search.search(run, max_evals=100)

    search = CBO(
        problem,
        acq_optimizer_kwargs=SEARCH_KWARGS_DEFAULTS["acq_optimizer_kwargs"],
        random_state=SEARCH_KWARGS_DEFAULTS["random_state"],
        surrogate_model="DUMMY",
        log_dir=os.path.join(tmp_path, "search_1"),
        verbose=0,
    )
    search.fit_generative_model(os.path.join(tmp_path, "search_0", "results.csv"))
    results_1 = search.search(run, max_evals=10)
    assert (results_1["p:x_cat"] == "c").sum() > 5
    assert (results_1["p:x_int"] >= 7).sum() > 5
    assert (results_1["p:x_float"] >= 7).sum() > 5


def test_convert_to_skopt_space():
    from ConfigSpace import Categorical, ConfigurationSpace, Float, Integer

    from deephyper.hpo._cbo import convert_to_skopt_space

    n_samples = 10

    # Case 1: without conditions and forbidden clauses
    config_space = ConfigurationSpace(
        {
            "x_cat": Categorical("x_cat", items=["a", "b", "c"]),
            "x_float": Float("x_float", bounds=(0.0, 10.0)),
            "x_int": Integer("x_int", bounds=(0, 1)),
        }
    )
    space = convert_to_skopt_space(config_space)
    samples_0 = space.rvs(n_samples, random_state=42)
    samples_1 = space.rvs(n_samples, random_state=42)
    samples_2 = space.rvs(n_samples, random_state=43)
    samples_3 = space.rvs(n_samples)
    assert np.all(samples_0 == samples_1)
    assert np.any(samples_0 != samples_2)
    assert np.any(samples_0 != samples_3)

    # Case 2: with conditions and forbidden clauses
    from ConfigSpace import ForbiddenEqualsClause

    config_space = ConfigurationSpace(
        {
            "x_cat": Categorical("x_cat", items=["a", "b", "c"]),
            "x_float": Float("x_float", bounds=(0.0, 10.0)),
            "x_int": Integer("x_int", bounds=(0, 10)),
        }
    )
    config_space.add(ForbiddenEqualsClause(config_space["x_cat"], "c"))

    space = convert_to_skopt_space(config_space)
    samples_0 = space.rvs(n_samples, random_state=42)
    samples_1 = space.rvs(n_samples, random_state=42)
    samples_2 = space.rvs(n_samples, random_state=43)
    samples_3 = space.rvs(n_samples)
    assert np.all(samples_0 == samples_1)
    assert np.any(samples_0 != samples_2)
    assert np.any(samples_0 != samples_3)


def test_checkpoint_restart_v2(tmp_path):
    import shutil

    from deephyper.hpo import CBO, HpProblem

    problem = HpProblem()
    problem.add_hyperparameter((0, 10), "x_int")
    problem.add_hyperparameter((0.0, 10.0), "x_float")
    problem.add_hyperparameter(["a", "b", "c"], "x_cat")

    def run(job):
        objective = (
            job.parameters["x_int"] + job.parameters["x_float"] + ord(job.parameters["x_cat"])
        )
        return objective

    search_kwargs = dict(
        problem=problem,
        initial_points=[problem.default_configuration],
        random_state=SEARCH_KWARGS_DEFAULTS["random_state"],
        acq_optimizer=SEARCH_KWARGS_DEFAULTS["acq_optimizer"],
        acq_optimizer_kwargs=SEARCH_KWARGS_DEFAULTS["acq_optimizer_kwargs"],
    )

    def get_log_dir(name):
        return os.path.join(tmp_path, name)

    if os.path.exists(get_log_dir("log_dir")):
        shutil.rmtree(get_log_dir("log_dir"))

    def run_search():
        search = CBO(
            log_dir=get_log_dir("log_dir"),
            checkpoint_restart=True,
            **search_kwargs,
        )

        results = search.search(run, 10)
        return results

    results = run_search()
    assert len(results) == 10
    results = run_search()
    assert len(results) == 10 * 2
    results = run_search()
    assert len(results) == 10 * 3
    assert len(set(results["job_id"])) == 30


def test_cbo_with_constraint_fn():
    def run_fn(j):
        return j.parameters["x"] ** 2 + j.parameters["y"] ** 2

    pb = HpProblem()
    pb.add((0.0, 10.0), "x")
    pb.add((0.0, 10.0), "y")

    def constraint_fn(df):
        return df["x"] + df["y"] < 10.0

    pb.set_constraint_fn(constraint_fn)

    # Test 1: with optimization "sampling"
    search = CBO(
        pb,
        verbose=1,
        acq_optimizer="sampling",
        acq_optimizer_kwargs={"n_points": 100},
        checkpoint_history_to_csv=False,
    )
    results = search.search(
        run_fn,
        max_evals=25,
    )

    assert all(results["p:x"] + results["p:y"] < 10.0)

    # Test 2: with optimization "ga"
    search = CBO(
        pb,
        verbose=1,
        acq_optimizer="ga",
        acq_optimizer_kwargs={"n_points": 100},
        checkpoint_history_to_csv=False,
    )
    results = search.search(
        run_fn,
        max_evals=25,
    )

    assert all(results["p:x"] + results["p:y"] < 10.0)

    # Test 3: with optimization "mixedga"
    search = CBO(
        pb,
        verbose=1,
        acq_optimizer="mixedga",
        acq_optimizer_kwargs={"n_points": 100},
        checkpoint_history_to_csv=False,
    )
    results = search.search(
        run_fn,
        max_evals=25,
    )

    assert all(results["p:x"] + results["p:y"] < 10.0)


if __name__ == "__main__":
    # test_sample_types(".")
    # test_gp(".")
    # test_cbo_categorical_variable(".")
    # test_cbo_checkpoint_restart_moo_with_failures(".")
    # test_cbo_checkpoint_restart_with_failures(".")
    # test_cbo_checkpoint_restart_moo(".")
    # test_many_initial_points(".")
    # test_convert_to_skopt_space()
    # test_max_total_failures()
    test_checkpoint_restart_v2(".")

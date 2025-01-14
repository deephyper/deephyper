import os
import time

import numpy as np
import pytest

SEARCH_KWARGS_DEFAULTS = dict(
    n_points=100,
    random_state=42,
    surrogate_model="ET",
    surrogate_model_kwargs={"n_estimators": 25, "min_samples_split": 8},
)


def test_cbo_random_seed(tmp_path):
    import numpy as np
    from deephyper.evaluator import Evaluator
    from deephyper.hpo import CBO, HpProblem

    problem = HpProblem()
    problem.add_hyperparameter((0.0, 10.0), "x")

    async def run_single_objective(config):
        return config["x"]

    def create_evaluator():
        return Evaluator.create(run_single_objective, method="serial")

    search = CBO(
        problem,
        create_evaluator(),
        n_points=SEARCH_KWARGS_DEFAULTS["n_points"],
        random_state=SEARCH_KWARGS_DEFAULTS["random_state"],
        surrogate_model="DUMMY",
        log_dir=tmp_path,
    )

    res1 = search.search(max_evals=4)
    res1_array = res1[["p:x"]].to_numpy()

    search = CBO(
        problem,
        create_evaluator(),
        n_points=SEARCH_KWARGS_DEFAULTS["n_points"],
        random_state=SEARCH_KWARGS_DEFAULTS["random_state"],
        surrogate_model="DUMMY",
        log_dir=tmp_path,
    )
    res2 = search.search(max_evals=4)
    res2_array = res2[["p:x"]].to_numpy()

    assert np.array_equal(res1_array, res2_array)

    # test multi-objective
    async def run_multi_objective(config):
        return config["x"], config["x"]

    def create_evaluator():
        return Evaluator.create(run_multi_objective, method="serial")

    search = CBO(
        problem,
        create_evaluator(),
        n_points=SEARCH_KWARGS_DEFAULTS["n_points"],
        random_state=SEARCH_KWARGS_DEFAULTS["random_state"],
        surrogate_model="DUMMY",
        log_dir=tmp_path,
    )

    res1 = search.search(max_evals=4)
    res1_array = res1[["p:x"]].to_numpy()

    search = CBO(
        problem,
        create_evaluator(),
        n_points=SEARCH_KWARGS_DEFAULTS["n_points"],
        random_state=SEARCH_KWARGS_DEFAULTS["random_state"],
        surrogate_model="DUMMY",
        log_dir=tmp_path,
    )
    res2 = search.search(max_evals=4)
    res2_array = res2[["p:x"]].to_numpy()

    assert np.array_equal(res1_array, res2_array)


def test_sample_types(tmp_path):
    import numpy as np
    from deephyper.hpo import CBO, HpProblem

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
        run,
        n_initial_points=5,
        n_points=SEARCH_KWARGS_DEFAULTS["n_points"],
        random_state=SEARCH_KWARGS_DEFAULTS["random_state"],
        surrogate_model="DUMMY",
        log_dir=tmp_path,
        verbose=0,
    ).search(max_evals)
    assert len(results) == max_evals

    results = CBO(
        problem,
        run,
        n_initial_points=5,
        n_points=SEARCH_KWARGS_DEFAULTS["n_points"],
        random_state=SEARCH_KWARGS_DEFAULTS["random_state"],
        surrogate_model=SEARCH_KWARGS_DEFAULTS["surrogate_model"],
        surrogate_model_kwargs=SEARCH_KWARGS_DEFAULTS["surrogate_model_kwargs"],
        log_dir=tmp_path,
        verbose=0,
    ).search(max_evals)
    assert len(results) == max_evals


def test_sample_types_no_cat(tmp_path):
    import numpy as np
    from deephyper.hpo import CBO, HpProblem

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
        run,
        random_state=SEARCH_KWARGS_DEFAULTS["random_state"],
        n_points=SEARCH_KWARGS_DEFAULTS["n_points"],
        n_initial_points=5,
        surrogate_model="DUMMY",
        log_dir=tmp_path,
        verbose=0,
    ).search(max_evals)
    assert len(results) == max_evals

    results = CBO(
        problem,
        run,
        random_state=SEARCH_KWARGS_DEFAULTS["random_state"],
        n_points=SEARCH_KWARGS_DEFAULTS["n_points"],
        n_initial_points=5,
        surrogate_model=SEARCH_KWARGS_DEFAULTS["surrogate_model"],
        surrogate_model_kwargs=SEARCH_KWARGS_DEFAULTS["surrogate_model_kwargs"],
        log_dir=tmp_path,
        verbose=0,
    ).search(max_evals)
    assert len(results) == max_evals


def test_gp(tmp_path):
    from deephyper.evaluator import Evaluator
    from deephyper.hpo import CBO, HpProblem

    # test float hyperparameters
    problem = HpProblem()
    problem.add_hyperparameter((0.0, 10.0), "x")

    async def run(config):
        return config["x"]

    max_evals = 20
    results = CBO(
        problem,
        Evaluator.create(run, method="serial"),
        n_points=SEARCH_KWARGS_DEFAULTS["n_points"],
        random_state=SEARCH_KWARGS_DEFAULTS["random_state"],
        surrogate_model="GP",
        acq_func="UCB",
        log_dir=tmp_path,
    ).search(max_evals)
    assert len(results) == max_evals

    # test int hyperparameters
    problem = HpProblem()
    problem.add_hyperparameter((0, 10), "x")

    results = CBO(
        problem,
        Evaluator.create(run, method="serial"),
        n_points=SEARCH_KWARGS_DEFAULTS["n_points"],
        random_state=SEARCH_KWARGS_DEFAULTS["random_state"],
        surrogate_model="GP",
        acq_func="UCB",
        log_dir=tmp_path,
    ).search(max_evals)
    assert len(results) == max_evals

    # test categorical hyperparameters
    problem = HpProblem()
    problem.add_hyperparameter([f"{i}" for i in range(10)], "x")

    async def run_cast_output_int(config):
        return int(config["x"])

    results = CBO(
        problem,
        Evaluator.create(run_cast_output_int, method="serial"),
        n_points=SEARCH_KWARGS_DEFAULTS["n_points"],
        random_state=SEARCH_KWARGS_DEFAULTS["random_state"],
        surrogate_model="GP",
        acq_func="UCB",
        log_dir=tmp_path,
    ).search(max_evals)
    assert len(results) == max_evals


def test_sample_types_conditional(tmp_path):
    import ConfigSpace as cs
    import numpy as np
    from deephyper.hpo import CBO, HpProblem

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
        run,
        n_points=SEARCH_KWARGS_DEFAULTS["n_points"],
        random_state=SEARCH_KWARGS_DEFAULTS["random_state"],
        surrogate_model="DUMMY",
        log_dir=tmp_path,
        verbose=0,
    )
    results = search.search(10)
    assert len(results) == 10

    # Test search with ET surrogate model
    results = CBO(
        problem,
        run,
        n_points=SEARCH_KWARGS_DEFAULTS["n_points"],
        random_state=SEARCH_KWARGS_DEFAULTS["random_state"],
        n_initial_points=5,
        surrogate_model=SEARCH_KWARGS_DEFAULTS["surrogate_model"],
        surrogate_model_kwargs=SEARCH_KWARGS_DEFAULTS["surrogate_model_kwargs"],
        log_dir=tmp_path,
        verbose=1,
    ).search(20)
    assert len(results) == 20


def run_max_evals(job):
    config = job.parameters
    return config["x"]


def test_max_evals_strict(tmp_path):
    from deephyper.evaluator import Evaluator
    from deephyper.hpo import CBO, HpProblem

    problem = HpProblem()
    problem.add_hyperparameter((0.0, 10.0), "x")

    evaluator = Evaluator.create(run_max_evals, method="process", method_kwargs={"num_workers": 8})

    # Test Timeout with max_evals (this should be like an "max_evals or timeout" condition)
    search = CBO(
        problem,
        evaluator,
        n_points=SEARCH_KWARGS_DEFAULTS["n_points"],
        random_state=SEARCH_KWARGS_DEFAULTS["random_state"],
        surrogate_model="DUMMY",
        log_dir=tmp_path,
    )

    max_evals = 100
    results = search.search(max_evals, max_evals_strict=True)
    assert len(results) == max_evals


def test_initial_points(tmp_path):
    from deephyper.hpo import CBO, HpProblem

    problem = HpProblem()
    problem.add_hyperparameter((0.0, 10.0), "x")

    def run(config):
        return config["x"]

    search = CBO(
        problem,
        run,
        initial_points=[problem.default_configuration],
        n_points=SEARCH_KWARGS_DEFAULTS["n_points"],
        random_state=SEARCH_KWARGS_DEFAULTS["random_state"],
        surrogate_model="DUMMY",
        log_dir=tmp_path,
    )

    result = search.search(10)
    assert len(result) == 10
    assert result.loc[0, "objective"] == problem.default_configuration["x"]


def test_cbo_checkpoint_restart(tmp_path):
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
        evaluator=run,
        initial_points=[problem.default_configuration],
        random_state=SEARCH_KWARGS_DEFAULTS["random_state"],
        n_points=SEARCH_KWARGS_DEFAULTS["n_points"],
    )

    def get_log_dir(name):
        return os.path.join(tmp_path, name)

    # test pause-continue of the search
    search_a = CBO(
        surrogate_model="DUMMY",
        log_dir=get_log_dir("search_a"),
        **search_kwargs,
    )

    results_a = search_a.search(4)
    assert len(results_a) == 4

    new_results_a = search_a.search(6)
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
    new_results_b = search_b.search(6)
    assert len(new_results_b) == 6

    # test reloading of a checkpoint from a file
    search_c = CBO(
        surrogate_model=SEARCH_KWARGS_DEFAULTS["surrogate_model"],
        surrogate_model_kwargs=SEARCH_KWARGS_DEFAULTS["surrogate_model_kwargs"],
        log_dir=get_log_dir("search_c"),
        **search_kwargs,
    )
    search_c.fit_surrogate(os.path.join(get_log_dir("search_b"), "results.csv"))
    results_c = search_c.search(20)

    assert len(results_c) == 20


def test_cbo_checkpoint_restart_moo(tmp_path):
    from deephyper.hpo import CBO, HpProblem

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
        evaluator=run,
        initial_points=[problem.default_configuration],
        random_state=SEARCH_KWARGS_DEFAULTS["random_state"],
        n_points=SEARCH_KWARGS_DEFAULTS["n_points"],
    )

    def get_log_dir(name):
        return os.path.join(tmp_path, name)

    # test pause-continue of the search
    search_a = CBO(
        log_dir=get_log_dir("search_a"),
        surrogate_model="DUMMY",
        **search_kwargs,
    )

    results_a = search_a.search(4)
    assert len(results_a) == 4

    new_results_a = search_a.search(6)
    assert all(results_a["p:x_int"] == new_results_a.iloc[:4]["p:x_int"])
    assert len(new_results_a) == 10

    # test reloading of a checkpoint directly as dataframe
    search_b = CBO(
        log_dir=get_log_dir("search_b"),
        surrogate_model=SEARCH_KWARGS_DEFAULTS["surrogate_model"],
        surrogate_model_kwargs=SEARCH_KWARGS_DEFAULTS["surrogate_model_kwargs"],
        **search_kwargs,
    )

    search_b.fit_surrogate(results_a)
    new_results_b = search_b.search(6)
    assert len(new_results_b) == 6

    # test reloading of a checkpoint from a file
    search_c = CBO(
        log_dir=get_log_dir("search_c"),
        surrogate_model=SEARCH_KWARGS_DEFAULTS["surrogate_model"],
        surrogate_model_kwargs=SEARCH_KWARGS_DEFAULTS["surrogate_model_kwargs"],
        **search_kwargs,
    )
    search_c.fit_surrogate(os.path.join(get_log_dir("search_b"), "results.csv"))
    results_c = search_c.search(20)

    assert len(results_c) == 20


def test_cbo_checkpoint_restart_with_failures(tmp_path):
    from deephyper.hpo import CBO, HpProblem

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
        evaluator=run,
        initial_points=[problem.default_configuration],
        random_state=SEARCH_KWARGS_DEFAULTS["random_state"],
        n_points=SEARCH_KWARGS_DEFAULTS["n_points"],
    )

    # test pause-continue of the search
    search_a = CBO(
        log_dir=os.path.join(tmp_path, "search_a"),
        surrogate_model="DUMMY",
        **search_kwargs,
    )

    results_a = search_a.search(10)
    assert len(results_a) == 10

    new_results_a = search_a.search(10)
    assert len(new_results_a) == 20

    # test reloading of a checkpoint directly as dataframe
    search_b = CBO(
        log_dir=os.path.join(tmp_path, "search_b"),
        surrogate_model=SEARCH_KWARGS_DEFAULTS["surrogate_model"],
        surrogate_model_kwargs=SEARCH_KWARGS_DEFAULTS["surrogate_model_kwargs"],
        **search_kwargs,
    )

    search_b.fit_surrogate(results_a)
    new_results_b = search_b.search(20)
    assert len(new_results_b) == 20

    # test reloading of a checkpoint from a file
    search_c = CBO(
        log_dir=os.path.join(tmp_path, "search_c"),
        surrogate_model=SEARCH_KWARGS_DEFAULTS["surrogate_model"],
        surrogate_model_kwargs=SEARCH_KWARGS_DEFAULTS["surrogate_model_kwargs"],
        **search_kwargs,
    )
    search_c.fit_surrogate(os.path.join(tmp_path, "search_b", "results.csv"))
    results_c = search_c.search(20)

    assert len(results_c) == 20


def test_cbo_checkpoint_restart_moo_with_failures(tmp_path):
    from deephyper.hpo import CBO, HpProblem

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
        evaluator=run,
        initial_points=[problem.default_configuration],
        random_state=SEARCH_KWARGS_DEFAULTS["random_state"],
        surrogate_model="DUMMY",
        n_points=SEARCH_KWARGS_DEFAULTS["n_points"],
    )

    # test pause-continue of the search
    search_a = CBO(
        log_dir=os.path.join(tmp_path, "search_a"),
        **search_kwargs,
    )

    results_a = search_a.search(10)
    assert len(results_a) == 10

    new_results_a = search_a.search(10)
    assert len(new_results_a) == 20

    # test reloading of a checkpoint directly as dataframe
    search_b = CBO(
        log_dir=os.path.join(tmp_path, "search_b"),
        **search_kwargs,
    )

    search_b.fit_surrogate(results_a)
    new_results_b = search_b.search(20)
    assert len(new_results_b) == 20

    # test reloading of a checkpoint from a file
    search_c = CBO(
        log_dir=os.path.join(tmp_path, "search_c"),
        **search_kwargs,
    )
    search_c.fit_surrogate(os.path.join(tmp_path, "search_b", "results.csv"))
    results_c = search_c.search(20)

    assert len(results_c) == 20


def test_cbo_categorical_variable(tmp_path):
    from deephyper.evaluator import SerialEvaluator
    from deephyper.hpo import CBO, HpProblem

    problem = HpProblem()
    problem.add_hyperparameter([32, 64, 96], "x", default_value=64)
    problem.add_hyperparameter((0.0, 10.0), "y", default_value=5.0)

    async def run(config):
        return config["x"] + config["y"]

    # test pause-continue of the search
    search = CBO(
        problem,
        SerialEvaluator(run, callbacks=[]),
        initial_points=[problem.default_configuration],
        n_points=SEARCH_KWARGS_DEFAULTS["n_points"],
        random_state=SEARCH_KWARGS_DEFAULTS["random_state"],
        surrogate_model="RF",
        log_dir=tmp_path,
    )

    results = search.search(25)
    assert results.objective.max() >= 105


def test_cbo_multi_point_strategy(tmp_path):
    from deephyper.evaluator import Evaluator
    from deephyper.hpo import CBO, HpProblem

    problem = HpProblem()
    problem.add_hyperparameter((0.0, 10.0), "x")

    async def run(config):
        return config["x"]

    durations = []
    for multi_point_strategy in ["cl_min", "cl_mean", "cl_max", "qUCB", "qUCBd"]:
        t1 = time.time()
        search = CBO(
            problem,
            Evaluator.create(run, method="serial", method_kwargs={"num_workers": 5}),
            n_points=SEARCH_KWARGS_DEFAULTS["n_points"],
            random_state=SEARCH_KWARGS_DEFAULTS["random_state"],
            surrogate_model="ET",
            surrogate_model_kwargs={"n_estimators": 25, "min_samples_split": 8},
            multi_point_strategy=multi_point_strategy,
            log_dir=tmp_path,
        )
        max_evals = 25
        results = search.search(max_evals)
        durations.append(time.time() - t1)

        assert len(results) == max_evals

    assert all(durations[i] > durations[j] for i in range(3) for j in range(3, 5))

    durations = []
    for multi_point_strategy in ["cl_min", "cl_mean", "cl_max", "qUCB", "qUCBd"]:
        t1 = time.time()
        search = CBO(
            problem,
            Evaluator.create(run, method="serial", method_kwargs={"num_workers": 5}),
            n_points=SEARCH_KWARGS_DEFAULTS["n_points"],
            acq_func="EI",
            random_state=SEARCH_KWARGS_DEFAULTS["random_state"],
            surrogate_model="ET",
            surrogate_model_kwargs={"n_estimators": 25, "min_samples_split": 8},
            multi_point_strategy=multi_point_strategy,
            log_dir=tmp_path,
        )
        max_evals = 25
        results = search.search(25)
        durations.append(time.time() - t1)

        assert len(results) == 25

    assert all(durations[i] > durations[j] for i in range(3) for j in range(3, 5))


@pytest.mark.slow
def test_cbo_with_acq_optimizer_mixedga_and_conditions_in_problem(tmp_path):
    from ConfigSpace import GreaterThanCondition
    from deephyper.hpo import CBO, HpProblem

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
    from ConfigSpace import ForbiddenAndConjunction, ForbiddenEqualsClause, ForbiddenEqualsRelation
    from deephyper.hpo import CBO, HpProblem

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


@pytest.mark.sdv
def test_cbo_fit_generative_model(tmp_path):
    from deephyper.hpo import CBO, HpProblem

    problem = HpProblem()
    problem.add_hyperparameter((0, 10), "x_int")
    problem.add_hyperparameter((0.0, 10.0), "x_float")
    problem.add_hyperparameter(["a", "b", "c"], "x_cat")

    def run(job):
        return sum(ord(v) if isinstance(v, str) else v for v in job.parameters.values())

    search = CBO(
        problem,
        run,
        n_points=SEARCH_KWARGS_DEFAULTS["n_points"],
        random_state=SEARCH_KWARGS_DEFAULTS["random_state"],
        surrogate_model="DUMMY",
        log_dir=os.path.join(tmp_path, "search_0"),
        verbose=0,
    )
    search.search(max_evals=100)

    search = CBO(
        problem,
        run,
        n_points=SEARCH_KWARGS_DEFAULTS["n_points"],
        random_state=SEARCH_KWARGS_DEFAULTS["random_state"],
        surrogate_model="DUMMY",
        log_dir=os.path.join(tmp_path, "search_1"),
        verbose=0,
    )
    search.fit_generative_model(os.path.join(tmp_path, "search_0", "results.csv"))
    results_1 = search.search(max_evals=10)
    assert (results_1["p:x_cat"] == "c").sum() > 5
    assert (results_1["p:x_int"] >= 7).sum() > 5
    assert (results_1["p:x_float"] >= 7).sum() > 5


if __name__ == "__main__":
    import logging

    logging.basicConfig(
        # filename=path_log_file, # optional if we want to store the logs to disk
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s - %(message)s",
        force=True,
    )

    tmp_path = "/tmp/deephyper_test"

    # test_cbo_with_acq_optimizer_mixedga_and_conditions_in_problem(tmp_path)
    test_cbo_with_acq_optimizer_mixedga_and_forbiddens_in_problem(tmp_path)

    # import time

    # # scope = locals().copy()
    # # # for k in scope:
    # # for k in [
    # #     "test_timeout",
    # #     # "test_max_evals_strict",
    # # ]:
    # #     if k.startswith("test_"):

    # #         print(f"Running {k}")

    # #         test_func = scope[k]

    # #         t1 = time.time()
    # #         test_func(tmp_path)
    # #         duration = time.time() - t1

    # #         print(f"\t{duration:.2f} sec.")

    # t1 = time.time()
    # # test_max_evals_strict(tmp_path)
    # test_timeout(tmp_path)
    # duration = time.time() - t1
    # print(f"\t{duration:.2f} sec.")

import pytest


@pytest.mark.hps
def test_cbo_random_seed(tmp_path):
    import numpy as np
    from deephyper.evaluator import Evaluator
    from deephyper.problem import HpProblem
    from deephyper.search.hps import CBO

    problem = HpProblem()
    problem.add_hyperparameter((0.0, 10.0), "x")

    def run(config):
        return config["x"]

    create_evaluator = lambda: Evaluator.create(run, method="serial")

    search = CBO(
        problem,
        create_evaluator(),
        random_state=42,
        surrogate_model="DUMMY",
        log_dir=tmp_path,
    )

    res1 = search.search(max_evals=4)
    res1_array = res1[["p:x"]].to_numpy()

    search = CBO(
        problem,
        create_evaluator(),
        random_state=42,
        surrogate_model="DUMMY",
        log_dir=tmp_path,
    )
    res2 = search.search(max_evals=4)
    res2_array = res2[["p:x"]].to_numpy()

    assert np.array_equal(res1_array, res2_array)

    # test multi-objective
    def run(config):
        return config["x"], config["x"]

    create_evaluator = lambda: Evaluator.create(run, method="serial")

    search = CBO(
        problem,
        create_evaluator(),
        random_state=42,
        surrogate_model="DUMMY",
        log_dir=tmp_path,
    )

    res1 = search.search(max_evals=4)
    res1_array = res1[["p:x"]].to_numpy()

    search = CBO(
        problem,
        create_evaluator(),
        random_state=42,
        surrogate_model="DUMMY",
        log_dir=tmp_path,
    )
    res2 = search.search(max_evals=4)
    res2_array = res2[["p:x"]].to_numpy()

    assert np.array_equal(res1_array, res2_array)


@pytest.mark.hps
def test_sample_types(tmp_path):
    import numpy as np
    from deephyper.evaluator import Evaluator
    from deephyper.problem import HpProblem
    from deephyper.search.hps import CBO

    problem = HpProblem()
    problem.add_hyperparameter((0, 10), "x_int")
    problem.add_hyperparameter((0.0, 10.0), "x_float")
    problem.add_hyperparameter([0, "1", 2.0], "x_cat")

    def run(config):

        assert np.issubdtype(type(config["x_int"]), np.integer)
        assert np.issubdtype(type(config["x_float"]), float)

        if config["x_cat"] == 0:
            assert np.issubdtype(type(config["x_cat"]), np.integer)
        elif config["x_cat"] == "1":
            assert type(config["x_cat"]) is str or type(config["x_cat"]) is np.str_
        else:
            assert np.issubdtype(type(config["x_cat"]), float)

        return 0

    create_evaluator = lambda: Evaluator.create(run, method="serial")

    results = CBO(
        problem,
        create_evaluator(),
        random_state=42,
        surrogate_model="DUMMY",
        log_dir=tmp_path,
    ).search(10)

    results = CBO(
        problem,
        create_evaluator(),
        random_state=42,
        surrogate_model="RF",
        log_dir=tmp_path,
    ).search(10)


@pytest.mark.hps
def test_sample_types_no_cat(tmp_path):
    import numpy as np
    from deephyper.evaluator import Evaluator
    from deephyper.problem import HpProblem
    from deephyper.search.hps import CBO

    problem = HpProblem()
    problem.add_hyperparameter((0, 10), "x_int")
    problem.add_hyperparameter((0.0, 10.0), "x_float")

    def run(config):

        assert np.issubdtype(type(config["x_int"]), np.integer)
        assert np.issubdtype(type(config["x_float"]), float)

        return 0

    create_evaluator = lambda: Evaluator.create(run, method="serial")

    results = CBO(
        problem,
        create_evaluator(),
        random_state=42,
        surrogate_model="DUMMY",
        log_dir=tmp_path,
    ).search(10)

    results = CBO(
        problem,
        create_evaluator(),
        random_state=42,
        surrogate_model="RF",
        log_dir=tmp_path,
    ).search(10)


@pytest.mark.hps
def test_gp(tmp_path):
    from deephyper.evaluator import Evaluator
    from deephyper.problem import HpProblem
    from deephyper.search.hps import CBO

    # test float hyperparameters
    problem = HpProblem()
    problem.add_hyperparameter((0.0, 10.0), "x")

    def run(config):
        return config["x"]

    results = CBO(
        problem,
        Evaluator.create(run, method="serial"),
        random_state=42,
        surrogate_model="GP",
        log_dir=tmp_path,
    ).search(10)

    # test int hyperparameters
    problem = HpProblem()
    problem.add_hyperparameter((0, 10), "x")

    def run(config):
        return config["x"]

    results = CBO(
        problem,
        Evaluator.create(run, method="serial"),
        random_state=42,
        surrogate_model="GP",
        log_dir=tmp_path,
    ).search(10)

    # test categorical hyperparameters
    problem = HpProblem()
    problem.add_hyperparameter([f"{i}" for i in range(10)], "x")

    def run(config):
        return int(config["x"])

    results = CBO(
        problem,
        Evaluator.create(run, method="serial"),
        random_state=42,
        surrogate_model="GP",
        log_dir=tmp_path,
    ).search(10)


@pytest.mark.hps
def test_sample_types_conditional(tmp_path):
    import ConfigSpace as cs
    import numpy as np
    from deephyper.evaluator import Evaluator
    from deephyper.problem import HpProblem
    from deephyper.search.hps import CBO

    problem = HpProblem()

    # choices
    choice = problem.add_hyperparameter(
        name="choice",
        value=["choice1", "choice2"],
    )

    # integers
    x1_int = problem.add_hyperparameter(name="x1_int", value=(1, 10))

    x2_int = problem.add_hyperparameter(name="x2_int", value=(1, 10))

    # conditions
    cond_1 = cs.EqualsCondition(x1_int, choice, "choice1")

    cond_2 = cs.EqualsCondition(x2_int, choice, "choice2")

    problem.add_condition(cond_1)
    problem.add_condition(cond_2)

    def run(config):

        if config["choice"] == "choice1":
            assert np.issubdtype(type(config["x1_int"]), np.integer)
        else:
            assert np.issubdtype(type(config["x2_int"]), np.integer)

        return 0

    create_evaluator = lambda: Evaluator.create(run, method="serial")

    results = CBO(
        problem,
        create_evaluator(),
        random_state=42,
        surrogate_model="DUMMY",
        log_dir=tmp_path,
    ).search(10)


@pytest.mark.hps
def test_timeout(tmp_path):
    import time
    from deephyper.problem import HpProblem
    from deephyper.search.hps import CBO

    problem = HpProblem()
    problem.add_hyperparameter((0.0, 10.0), "x")

    def run(config):
        try:
            # simulate working thread
            while True:
                1 + 1
        except:  # simulate the catching of any exception here
            time.sleep(2)
        return config["x"]

    search = CBO(
        problem, run, random_state=42, surrogate_model="DUMMY", log_dir=tmp_path
    )

    t1 = time.time()
    result = search.search(timeout=1)
    duration = time.time() - t1
    assert duration < 1.5


@pytest.mark.hps
def test_initial_points(tmp_path):
    from deephyper.problem import HpProblem
    from deephyper.search.hps import CBO

    problem = HpProblem()
    problem.add_hyperparameter((0.0, 10.0), "x")

    def run(config):
        return config["x"]

    search = CBO(
        problem,
        run,
        initial_points=[problem.default_configuration],
        random_state=42,
        surrogate_model="DUMMY",
        log_dir=tmp_path,
    )

    result = search.search(10)
    assert len(result) == 10
    assert result.loc[0, "objective"] == problem.default_configuration["x"]


@pytest.mark.hps
def test_cbo_checkpoint_restart(tmp_path):
    from deephyper.problem import HpProblem
    from deephyper.search.hps import CBO

    problem = HpProblem()
    problem.add_hyperparameter((0.0, 10.0), "x")

    def run(config):
        return config["x"]

    # test pause-continue of the search
    search_a = CBO(
        problem,
        run,
        initial_points=[problem.default_configuration],
        random_state=42,
        surrogate_model="DUMMY",
        log_dir=tmp_path,
    )

    results_a = search_a.search(4)
    assert len(results_a) == 4

    new_results_a = search_a.search(6)
    assert len(new_results_a) == 10

    # test reloading of a checkpoint
    search_b = CBO(
        problem,
        run,
        initial_points=[problem.default_configuration],
        random_state=42,
        surrogate_model="DUMMY",
        log_dir=tmp_path,
    )

    search_b.fit_surrogate(results_a)
    new_results_b = search_b.search(6)
    assert len(new_results_b) == 6


if __name__ == "__main__":
    test_cbo_checkpoint_restart(tmp_path=".")

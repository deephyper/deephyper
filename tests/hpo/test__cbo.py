import pytest


@pytest.mark.hps
def test_cbo_random_seed(tmp_path):
    import numpy as np
    from deephyper.evaluator import Evaluator
    from deephyper.hpo import HpProblem
    from deephyper.hpo import CBO

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
    from deephyper.hpo import HpProblem
    from deephyper.hpo import CBO

    problem = HpProblem()
    problem.add_hyperparameter((0, 10), "x_int")
    problem.add_hyperparameter((0.0, 10.0), "x_float")
    problem.add_hyperparameter(["0", "1", "2"], "x_cat")

    def run(config):
        assert np.issubdtype(type(config["x_int"]), np.integer)
        assert np.issubdtype(type(config["x_float"]), float)
        assert type(config["x_cat"]) is str or type(config["x_cat"]) is np.str_

        return config["x_int"] + config["x_float"] + int(config["x_cat"])

    results = CBO(
        problem,
        run,
        n_initial_points=5,
        random_state=42,
        surrogate_model="DUMMY",
        log_dir=tmp_path,
        verbose=0,
    ).search(10)

    results = CBO(
        problem,
        run,
        n_initial_points=5,
        random_state=42,
        surrogate_model="RF",
        log_dir=tmp_path,
        verbose=0,
    ).search(10)


@pytest.mark.hps
def test_sample_types_no_cat(tmp_path):
    import numpy as np
    from deephyper.evaluator import Evaluator
    from deephyper.hpo import HpProblem
    from deephyper.hpo import CBO

    problem = HpProblem()
    problem.add_hyperparameter((0, 10), "x_int")
    problem.add_hyperparameter((0.0, 10.0), "x_float")

    def run(config):
        assert np.issubdtype(type(config["x_int"]), np.integer)
        assert np.issubdtype(type(config["x_float"]), float)

        return config["x_int"] + config["x_float"]

    results = CBO(
        problem,
        run,
        random_state=42,
        n_initial_points=5,
        surrogate_model="DUMMY",
        log_dir=tmp_path,
        verbose=0,
    ).search(10)

    results = CBO(
        problem,
        run,
        random_state=42,
        n_initial_points=5,
        surrogate_model="RF",
        log_dir=tmp_path,
        verbose=0,
    ).search(10)


@pytest.mark.hps
def test_gp(tmp_path):
    from deephyper.evaluator import Evaluator
    from deephyper.hpo import HpProblem
    from deephyper.hpo import CBO

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
        acq_func="UCB",
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
        acq_func="UCB",
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
        acq_func="UCB",
        log_dir=tmp_path,
    ).search(10)


@pytest.mark.hps
def test_sample_types_conditional(tmp_path):
    import ConfigSpace as cs
    import numpy as np
    from deephyper.evaluator import Evaluator
    from deephyper.hpo import HpProblem
    from deephyper.hpo import CBO

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
        random_state=42,
        surrogate_model="DUMMY",
        log_dir=tmp_path,
        verbose=0,
    )
    results = search.search(10)
    assert len(results) == 10

    # # Test search with transfer learning through generative model
    # search = CBO(
    #     problem,
    #     run,
    #     random_state=42,
    #     surrogate_model="DUMMY",
    #     log_dir=tmp_path,
    #     verbose=0,
    # )
    # search.fit_generative_model(results)
    # results = search.search(10)
    # assert len(results) == 10

    # Test search with ET surrogate model
    results = CBO(
        problem,
        run,
        random_state=42,
        n_initial_points=5,
        surrogate_model="ET",
        log_dir=tmp_path,
        verbose=1,
    ).search(20)
    assert len(results) == 20


@pytest.mark.hps
def test_timeout(tmp_path):
    import time
    from deephyper.hpo import HpProblem
    from deephyper.hpo import CBO

    problem = HpProblem()
    problem.add_hyperparameter((0.0, 10.0), "x")

    def run(job):
        config = job.parameters
        print("job:", job.id)
        try:
            # simulate working thread
            while True:
                1 + 1
        except:  # simulate the catching of any exception here
            time.sleep(2)
        return config["x"]

    # Test Timeout without max_evals
    search = CBO(
        problem, run, random_state=42, surrogate_model="DUMMY", log_dir=tmp_path
    )

    t1 = time.time()
    result = search.search(timeout=1)
    duration = time.time() - t1
    assert duration < 3
    assert result is None

    # Test Timeout with max_evals (this should be like an "max_evals or timeout" condition)
    search = CBO(
        problem, run, random_state=42, surrogate_model="DUMMY", log_dir=tmp_path
    )

    t1 = time.time()
    result = search.search(max_evals=10, timeout=1, max_evals_strict=True)
    duration = time.time() - t1
    assert duration < 3
    assert result is None


def run_max_evals(job):
    config = job.parameters
    return config["x"]


@pytest.mark.hps
def test_max_evals_strict(tmp_path):
    from deephyper.hpo import HpProblem
    from deephyper.hpo import CBO
    from deephyper.evaluator import Evaluator

    problem = HpProblem()
    problem.add_hyperparameter((0.0, 10.0), "x")

    evaluator = Evaluator.create(
        run_max_evals, method="process", method_kwargs={"num_workers": 8}
    )

    # Test Timeout with max_evals (this should be like an "max_evals or timeout" condition)
    search = CBO(
        problem, evaluator, random_state=42, surrogate_model="DUMMY", log_dir=tmp_path
    )

    max_evals = 100
    results = search.search(max_evals, max_evals_strict=True)
    assert len(results) == max_evals


@pytest.mark.hps
def test_initial_points(tmp_path):
    from deephyper.hpo import HpProblem
    from deephyper.hpo import CBO

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
    from deephyper.hpo import HpProblem
    from deephyper.hpo import CBO

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
    assert len(new_results_a) == 6

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


@pytest.mark.hps
def test_cbo_categorical_variable(tmp_path):
    from deephyper.hpo import HpProblem
    from deephyper.hpo import CBO
    from deephyper.evaluator import SerialEvaluator

    problem = HpProblem()
    problem.add_hyperparameter([32, 64, 96], "x", default_value=64)
    problem.add_hyperparameter((0.0, 10.0), "y", default_value=5.0)

    def run(config):
        return config["x"] + config["y"]

    # test pause-continue of the search
    search = CBO(
        problem,
        SerialEvaluator(run, callbacks=[]),
        initial_points=[problem.default_configuration],
        random_state=42,
        surrogate_model="RF",
        log_dir=tmp_path,
    )

    results = search.search(25)
    assert results.objective.max() >= 105


@pytest.mark.slow
@pytest.mark.hps
def test_cbo_with_acq_optimizer_mixedga_and_conditions_in_problem(tmp_path):
    import numpy as np
    from deephyper.hpo import HpProblem
    from deephyper.hpo import CBO
    from deephyper.evaluator import Evaluator

    from ConfigSpace import GreaterThanCondition

    problem = HpProblem()

    max_num_layers = 10
    num_layers = problem.add_hyperparameter(
        (1, max_num_layers), "num_layers", default_value=2
    )

    conditions = []
    for i in range(max_num_layers):
        layer_i_units = problem.add_hyperparameter(
            (1, 100), f"layer_{i}_units", default_value=32
        )

        if i > 0:
            conditions.extend(
                [
                    GreaterThanCondition(layer_i_units, num_layers, i),
                ]
            )

    problem.add_conditions(conditions)
    print(problem)

    def run(job):
        num_layers = job.parameters["num_layers"]
        return sum(job.parameters[f"layer_{i}_units"] for i in range(num_layers))

    search = CBO(
        problem,
        run,
        random_state=42,
        verbose=1,
        log_dir=tmp_path,
        acq_optimizer="mixedga",
        acq_optimizer_freq=1,
        kappa=5.0,
        scheduler={"type": "periodic-exp-decay", "period": 25, "kappa_final": 0.0001},
        objective_scaler="identity",
    )
    results = search.search(max_evals=100)

    import matplotlib.pyplot as plt
    from deephyper.analysis.hpo import plot_search_trajectory_single_objective_hpo

    fig, ax = plot_search_trajectory_single_objective_hpo(results)
    plt.show()


# @pytest.mark.slow
# @pytest.mark.hps
# def test_cbo_with_acq_optimizer_mixedga_and_forbiddens_in_problem(tmp_path):
#     # TODO: this does not enforce the forbidden contraints
#     import numpy as np
#     from deephyper.hpo import HpProblem
#     from deephyper.hpo import CBO
#     from deephyper.evaluator import Evaluator

#     from ConfigSpace import ForbiddenEqualsRelation

#     problem = HpProblem()

#     max_num_layers = 2

#     for i in range(max_num_layers):
#         problem.add_hyperparameter((1, 5), f"layer_{i}_units")

#     forbiddens = []
#     for i in range(1, max_num_layers):
#         forb = ForbiddenEqualsRelation(
#             problem.space[f"layer_{i-1}_units"], problem.space[f"layer_{i}_units"]
#         )
#         forbiddens.append(forb)

#     print(problem)

#     def run(job):
#         return sum(job.parameters[f"layer_{i}_units"] for i in range(max_num_layers))

#     search = CBO(
#         problem,
#         run,
#         random_state=42,
#         verbose=1,
#         log_dir=tmp_path,
#         acq_optimizer="mixedga",
#         acq_optimizer_freq=1,
#         kappa=5.0,
#         scheduler={"type": "periodic-exp-decay", "period": 25, "kappa_final": 0.0001},
#         objective_scaler="identity",
#     )
#     results = search.search(max_evals=100)

#     import matplotlib.pyplot as plt
#     from deephyper.analysis.hpo import plot_search_trajectory_single_objective_hpo

#     fig, ax = plot_search_trajectory_single_objective_hpo(results)
#     plt.show()


if __name__ == "__main__":
    tmp_path = "/tmp/deephyper_test"
    # test_cbo_random_seed(tmp_path)
    # test_sample_types(tmp_path)
    # test_sample_types_conditional(tmp_path)
    # test_timeout(tmp_path)
    # test_cbo_with_acq_optimizer_mixedga_and_forbiddens_in_problem(tmp_path)
    # test_gp(tmp_path)
    # test_cbo_checkpoint_restart(tmp_path)
    test_max_evals_strict(tmp_path)
    # test_initial_points(tmp_path)

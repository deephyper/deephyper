import unittest

import pytest


@pytest.mark.hps_fast_test
class CBOTest(unittest.TestCase):
    def test_random_seed(self):
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
            problem, create_evaluator(), random_state=42, surrogate_model="DUMMY"
        )

        res1 = search.search(max_evals=4)
        res1_array = res1[["x"]].to_numpy()

        search = CBO(
            problem, create_evaluator(), random_state=42, surrogate_model="DUMMY"
        )
        res2 = search.search(max_evals=4)
        res2_array = res2[["x"]].to_numpy()

        assert np.array_equal(res1_array, res2_array)

    def test_sample_types(self):
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

        CBO(
            problem, create_evaluator(), random_state=42, surrogate_model="DUMMY"
        ).search(10)

        CBO(problem, create_evaluator(), random_state=42, surrogate_model="RF").search(
            10
        )

    def test_sample_types_no_cat(self):
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

        CBO(
            problem, create_evaluator(), random_state=42, surrogate_model="DUMMY"
        ).search(10)

        CBO(problem, create_evaluator(), random_state=42, surrogate_model="RF").search(
            10
        )

    def test_gp(self):
        from deephyper.evaluator import Evaluator
        from deephyper.problem import HpProblem
        from deephyper.search.hps import CBO

        # test float hyperparameters
        problem = HpProblem()
        problem.add_hyperparameter((0.0, 10.0), "x")

        def run(config):
            return config["x"]

        CBO(
            problem,
            Evaluator.create(run, method="serial"),
            random_state=42,
            surrogate_model="GP",
        ).search(10)

        # test int hyperparameters
        problem = HpProblem()
        problem.add_hyperparameter((0, 10), "x")

        def run(config):
            return config["x"]

        CBO(
            problem,
            Evaluator.create(run, method="serial"),
            random_state=42,
            surrogate_model="GP",
        ).search(10)

        # test categorical hyperparameters
        problem = HpProblem()
        problem.add_hyperparameter([f"{i}" for i in range(10)], "x")

        def run(config):
            return int(config["x"])

        CBO(
            problem,
            Evaluator.create(run, method="serial"),
            random_state=42,
            surrogate_model="GP",
        ).search(10)

    def test_sample_types_conditional(self):
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

        CBO(
            problem, create_evaluator(), random_state=42, surrogate_model="DUMMY"
        ).search(10)

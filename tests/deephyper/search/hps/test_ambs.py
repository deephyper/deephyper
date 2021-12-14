import unittest

import ConfigSpace as cs
import numpy as np
from deephyper.evaluator import Evaluator
from deephyper.problem import HpProblem
from deephyper.search.hps import AMBS


class AMBSTest(unittest.TestCase):
    def test_random_seed(self):

        problem = HpProblem()
        problem.add_hyperparameter((0.0, 10.0), "x")

        def run(config):
            return config["x"]

        create_evaluator = lambda: Evaluator.create(run, method="serial")

        search = AMBS(
            problem, create_evaluator(), random_state=42, surrogate_model="DUMMY"
        )

        res1 = search.search(max_evals=4)
        res1_array = res1[["x"]].to_numpy()

        search = AMBS(
            problem, create_evaluator(), random_state=42, surrogate_model="DUMMY"
        )
        res2 = search.search(max_evals=4)
        res2_array = res2[["x"]].to_numpy()

        assert np.array_equal(res1_array, res2_array)

    def test_sample_types(self):

        problem = HpProblem()
        problem.add_hyperparameter((0, 10), "x_int")
        problem.add_hyperparameter((0.0, 10.0), "x_float")
        problem.add_hyperparameter([0, "1", 2.0], "x_cat")

        def run(config):

            print(config)

            assert np.issubdtype(type(config["x_int"]), np.integer)
            assert np.issubdtype(type(config["x_float"]), np.float)

            if config["x_cat"] == 0:
                assert np.issubdtype(type(config["x_cat"]), np.integer)
            elif config["x_cat"] == "1":
                assert type(config["x_cat"]) is str or type(config["x_cat"]) is np.str_
            else:
                assert np.issubdtype(type(config["x_cat"]), np.float)

            return 0

        create_evaluator = lambda: Evaluator.create(run, method="serial")

        AMBS(
            problem, create_evaluator(), random_state=42, surrogate_model="DUMMY"
        ).search(10)

        AMBS(problem, create_evaluator(), random_state=42, surrogate_model="RF").search(
            10
        )

    def test_gp(self):

        # test float hyperparameters
        problem = HpProblem()
        problem.add_hyperparameter((0.0, 10.0), "x")

        def run(config):
            return config["x"]

        AMBS(
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

        AMBS(
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

        AMBS(
            problem,
            Evaluator.create(run, method="serial"),
            random_state=42,
            surrogate_model="GP",
        ).search(10)

    def test_conditional_sample_types(self):

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

            print(f"x1_int: {type(config['x1_int'])}")
            print(f"x2_int: {type(config['x2_int'])}")

            if config["choice"] == "choice1":
                assert np.issubdtype(type(config["x1_int"]), np.integer)
            else:
                assert np.issubdtype(type(config["x2_int"]), np.integer)

            return 0

        create_evaluator = lambda: Evaluator.create(run, method="serial")

        AMBS(
            problem, create_evaluator(), random_state=42, surrogate_model="DUMMY"
        ).search(10)

import unittest

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

import pytest
import numpy as np
from deephyper.problem import HpProblem
from deephyper.evaluator import Evaluator
from deephyper.search.hps import AMBS


def run(config):
    return config["x"]


problem = HpProblem()
problem.add_hyperparameter((0.0, 10.0), "x")


def test_ambs():

    create_evaluator = lambda: Evaluator.create(
        run, method="process", method_kwargs={"num_workers": 1}
    )

    search = AMBS(
        problem,
        create_evaluator(),
        random_state=42,
    )

    res1 = search.search(max_evals=4)
    res1_array = res1[["x"]].to_numpy()

    search.search(max_evals=100, timeout=1)

    search = AMBS(
        problem,
        create_evaluator(),
        random_state=42,
    )
    res2 = search.search(max_evals=4)
    res2_array = res2[["x"]].to_numpy()

    assert np.array_equal(res1_array, res2_array)


if __name__ == "__main__":
    test_ambs()

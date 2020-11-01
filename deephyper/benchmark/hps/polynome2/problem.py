"""
python -m deephyper.search.hps.ambs2 --evaluator threadPool --problem deephyper.benchmark.hps.polynome2.Problem --run deephyper.benchmark.hps.polynome2.run --max-evals 100 --kappa 0.001
"""
import os

import numpy as np

from deephyper.benchmark.benchmark_functions_wrappers import polynome_2
from deephyper.problem.base import BaseProblem

# Problem definition
Problem = BaseProblem()

num_dim = 10
for i in range(num_dim):
    Problem.add_dim(f"e{i}", (-10.0, 10.0))

Problem.add_starting_point(**{f"e{i}": 10.0 for i in range(num_dim)})

# Definition of the function which runs the model


def run(param_dict):
    f, _, _ = polynome_2()

    num_dim = 10
    x = np.array([param_dict[f"e{i}"] for i in range(num_dim)])

    return f(x)  # the objective


if __name__ == "__main__":
    print(Problem)

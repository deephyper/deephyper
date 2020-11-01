"""
!! CURRENT OPTIMIZER DOES NOT SUPPORT THIS!!

Hyperparameter optimization problem to try forbidden clauses.

Example command line::

    python -m deephyper.search.hps.ambs2 --evaluator threadPool --problem deephyper.benchmark.hps.toy.problem_basic_1.Problem --run deephyper.benchmark.hps.toy.problem_forbidden_1.run --max-evals 100 --kappa 0.001
"""
import numpy as np

from deephyper.problem import BaseProblem
from deephyper.problem import config_space as cs

# Problem definition
Problem = BaseProblem()

x_hp = Problem.add_hyperparameter(
    name="x", value=(0, 10)
)  # or Problem.add_dim("x", (0, 10))
y_hp = Problem.add_hyperparameter(
    name="y", value=(0, 10)
)  # or Problem.add_dim("y", (0, 10))

not_zero = cs.ForbiddenAndConjunction(
    cs.ForbiddenEqualsClause(x_hp, 0), cs.ForbiddenEqualsClause(y_hp, 0)
)

Problem.add_forbidden_clause(not_zero)

Problem.add_starting_point(x=1, y=1)

# Definition of the function which runs the model


def run(param_dict):

    x = param_dict["x"]
    y = param_dict["y"]

    res = np.log(1 / (y + x))

    return res  # the objective


if __name__ == "__main__":
    print(Problem)

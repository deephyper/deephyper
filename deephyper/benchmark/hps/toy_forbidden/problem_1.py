import numpy as np

from deephyper.problem import config_space as cs
from deephyper.problem import BaseProblem

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

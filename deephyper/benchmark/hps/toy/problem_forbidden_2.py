"""
!! CURRENT OPTIMIZER DOES NOT SUPPORT THIS!!

Hyperparameter optimization problem to try forbidden clauses by directly using ConfigSpace.

Example command line::

    python -m deephyper.search.hps.ambs2 --evaluator threadPool --problem deephyper.benchmark.hps.toy.problem_basic_1.Problem --run deephyper.benchmark.hps.toy.problem_forbidden_2.run --max-evals 100 --kappa 0.001
"""
import ConfigSpace as cs
import ConfigSpace.hyperparameters as csh
import numpy as np

from deephyper.problem import BaseProblem

config_space = cs.ConfigurationSpace(seed=42)

x_hp = csh.UniformIntegerHyperparameter(name="x", lower=0, upper=10, log=False)
y_hp = csh.UniformIntegerHyperparameter(name="y", lower=0, upper=10, log=False)

config_space.add_hyperparameters([x_hp, y_hp])

not_zero = cs.ForbiddenAndConjunction(
    cs.ForbiddenEqualsClause(x_hp, 0), cs.ForbiddenEqualsClause(y_hp, 0)
)

config_space.add_forbidden_clause(not_zero)

# Problem definition
Problem = BaseProblem(config_space)

Problem.add_starting_point(x=1, y=1)

# Definition of the function which runs the model


def run(param_dict):

    x = param_dict["x"]
    y = param_dict["y"]

    res = np.log(1 / (y + x))

    return res  # the objective


if __name__ == "__main__":
    print(Problem)

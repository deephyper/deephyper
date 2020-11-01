"""
Hyperparameter optimization problem to try forbidden clauses by directly using ConfigSpace.

Example command line::

    python -m deephyper.search.hps.ambsv1 --evaluator threadPool --problem deephyper.benchmark.hps.toy.problem_cond_1.Problem --run deephyper.benchmark.hps.toy.problem_cond_1.run --max-evals 100 --kappa 0.001
"""
import ConfigSpace as cs
from deephyper.problem import HpProblem

Problem = HpProblem(seed=45)

func = Problem.add_hyperparameter(name="func", value=["f1", "f2"])

# f1 variables
f1_x = Problem.add_hyperparameter(name="f1_x", value=(0.0, 1.0, "uniform"))
f1_y = Problem.add_hyperparameter(name="f1_y", value=(0.0, 1.0, "uniform"))

# f2 variables
f2_x = Problem.add_hyperparameter(name="f2_x", value=(0.0, 1.0, "uniform"))
f2_y = Problem.add_hyperparameter(name="f2_y", value=(0.0, 1.0, "uniform"))

cond_f1_x = cs.EqualsCondition(f1_x, func, "f1")
cond_f1_y = cs.EqualsCondition(f1_y, func, "f1")
Problem.add_condition(cond_f1_x)
Problem.add_condition(cond_f1_y)

cond_f2_x = cs.EqualsCondition(f2_x, func, "f2")
cond_f2_y = cs.EqualsCondition(f2_y, func, "f2")
Problem.add_condition(cond_f2_x)
Problem.add_condition(cond_f2_y)


# Definition of the function which runs the model
def run(param_dict):

    func = param_dict["func"]
    if func == "f1":
        return param_dict["f1_x"] + param_dict["f1_y"]
    else:
        return param_dict["f2_x"] * param_dict["f2_y"]


if __name__ == "__main__":
    print(Problem)

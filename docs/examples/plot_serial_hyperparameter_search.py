# -*- coding: utf-8 -*-
"""
Serial Hyperparameter optimization 
==================================

**Author(s)**: Romain Egele.

This example demonstrates how to run a serial execution (i.e., not parallel) of hyperparameter optimization. We start by defining the black-box function we want to optimize. For the purpose of simplicity we start by optimizing (maximising) the :math:`y = -x^2` function. This black-box function is often named ``run``-function inside DeepHyper. It takes as first argument an input-dictionnary ``config`` with a set of particular variable choices. It returns a single-scalar value which is maximised by DeepHyper.
"""
def run(config: dict) -> float:
    return -config["x"] ** 2


# %%
# Then we define the variable(s) we want to optimize. In our case we have a single variable :math:`x` and we # define a input space in the :math:`[-10,10]` continuous interval by providing boundaries with ``float`` 
# types.
from deephyper.problem import HpProblem


problem = HpProblem()
problem.add_hyperparameter((-10.0, 10.0), "x")

print(problem)

# %% 
# Then we define the evaluator which handles the execution of the black-box function. We decide to use the
# ``method="serial"`` which executes one run-function evaluation at a time without parallelism. We also
# provide the ``TqdmCallback`` to have an interactive feedback on the advancement of the search.
from deephyper.evaluator import Evaluator
from deephyper.evaluator.callback import TqdmCallback


max_evals = 25
evaluator = Evaluator.create(
    run, method="serial", method_kwargs={"callbacks": [TqdmCallback(max_evals)]}
)

# %%
# Then, we define a centralized Bayesian optimization (CBO) search (i.e., master-worker architecture) where we decide to use the Gaussian-Process regressor which is efficient for few-serial iterations of Bayesian optimization.
from deephyper.search.hps import CBO


search = CBO(problem, evaluator, surrogate_model="GP", random_state=42)
results = search.search(max_evals)

print(results)

# %%
# A Pandas DataFrame is returned by the search from which we can plot the results.
import matplotlib.pyplot as plt


plt.figure()
plt.scatter(list(range(max_evals)), results.objective)
plt.xlabel(r"Iterations")
plt.ylabel(r"$-x^2$")
plt.show()

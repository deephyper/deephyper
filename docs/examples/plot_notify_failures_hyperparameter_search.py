# -*- coding: utf-8 -*-
"""
Notify Failures in Hyperparameter optimization 
==============================================

**Author(s)**: Romain Egele.

This example demonstrates how to handle failure of objectives in hyperparameter search. In many cases such as software auto-tuning (where we minimize the run-time of a software application) some configurations can create run-time errors and therefore no scalar objective is returned. A default choice could be to return in this case the worst case objective if known and it can be done inside the ``run``-function. Other possibilites are to ignore these configurations or to replace them with the running mean/min objective. To illustrate such a use-case we define an artificial ``run``-function which will fail when one of its input parameters is greater than 0.5. To define a failure, it is possible to return a "string" value with ``"F"`` as prefix such as:
"""


def run(config: dict) -> float:
    if config["y"] > 0.5:
        return "F_postfix"
    else:
        return config["x"]


# %%
# Then, we define the corresponding hyperparameter problem where ``x`` is the value to maximize and ``y`` is a value impact the appearance of failures.
from deephyper.problem import HpProblem

problem = HpProblem()
problem.add_hyperparameter([1, 2, 4, 8, 16, 32], "x")
problem.add_hyperparameter((0.0, 1.0), "y")

print(problem)


# %%
# Then, we define a centralized Bayesian optimization (CBO) search (i.e., master-worker architecture) which uses the Random-Forest regressor as default surrogate model. We will compare the ``ignore`` strategy which filters-out failed configurations, the ``mean`` strategy which replaces a failure by the running mean of collected objectives and the ``min`` strategy which replaces by the running min of collected objectives.
from deephyper.search.hps import CBO
from deephyper.evaluator import Evaluator
from deephyper.evaluator.callback import TqdmCallback

results = {}
max_evals = 30
for failure_strategy in ["ignore", "mean", "min"]:
    # for failure_strategy in ["min"]:
    print(f"Executing failure strategy: {failure_strategy}")
    evaluator = Evaluator.create(
        run, method="serial", method_kwargs={"callbacks": [TqdmCallback(max_evals)]}
    )
    search = CBO(
        problem,
        evaluator,
        filter_failures=failure_strategy,
        log_dir=f"search_{failure_strategy}",
        random_state=42,
    )
    results[failure_strategy] = search.search(max_evals)

# %%
# Finally we plot the collected results
import matplotlib.pyplot as plt
import numpy as np

plt.figure()

for i, (failure_strategy, df) in enumerate(results.items()):
    plt.subplot(3, 1, i + 1)
    if df.objective.dtype != np.float64:
        x = np.arange(len(df))
        mask_failed = np.where(df.objective.str.startswith("F"))[0]
        mask_success = np.where(~df.objective.str.startswith("F"))[0]
        x_success, x_failed = x[mask_success], x[mask_failed]
        y_success = df["objective"][mask_success].astype(float)
    plt.scatter(x_success, y_success, label=failure_strategy)
    plt.scatter(x_failed, np.zeros(x_failed.shape), marker="v", color="red")

    plt.xlabel(r"Iterations")
    plt.ylabel(r"Objective")
    plt.legend()
plt.show()

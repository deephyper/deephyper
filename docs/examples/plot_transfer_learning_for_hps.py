# -*- coding: utf-8 -*-
"""
Transfer Learning for Hyperparameter Search
===========================================

**Author(s)**: Romain Egele.

In this example we present how to apply transfer-learning for hyperparameter search. Let's assume you have a bunch of similar tasks for example the search of neural networks hyperparameters for different datasets. You can easily imagine that close choices of hyperparameters can perform well these different datasets even if some light additional tuning can help improve the performance. Therefore, you can perform an expensive search once to then reuse the explored set of hyperparameters of thid search and bias the following search with it. Here, we will use a cheap to compute and easy to understand example where we maximise the :math:`f(x) = -\sum_{i=0}^{n-1}` function. In this case the size of the problem can be defined by the variable :math:`n`. We will start by optimizing the small-size problem where :math:`n=1`, then apply transfer-learning from to optimize the larger-size problem where :math:`n=2` and visualize the difference if were not to apply transfer-learning on this larger problem instance.

Let us start by defining the run-functions of the small and large scale problems:
"""

# %%
import functools


def run(config: dict, N: int) -> float:
    y = -sum([config[f"x{i}"] ** 2 for i in range(N)])
    return y


run_small = functools.partial(run, N=1)
run_large = functools.partial(run, N=2)

# %%
# Then, we can define the hyperparameter problem space based on :math:`n`
from deephyper.problem import HpProblem


N = 1
problem_small = HpProblem()
for i in range(N):
    problem_small.add_hyperparameter((-10.0, 10.0), f"x{i}")
problem_small

# %%
N = 2
problem_large = HpProblem()
for i in range(N):
    problem_large.add_hyperparameter((-10.0, 10.0), f"x{i}")
problem_large

# %%
# Then, we define setup the search and execute it:
from deephyper.evaluator import Evaluator
from deephyper.evaluator.callback import TqdmCallback
from deephyper.search.hps import CBO

results = {}
max_evals = 20
evaluator_small = Evaluator.create(
    run_small, method="serial", method_kwargs={"callbacks": [TqdmCallback(max_evals)]}
)
search_small = CBO(problem_small, evaluator_small, random_state=42)
results["Small"] = search_small.search(max_evals)

# %%
evaluator_large = Evaluator.create(
    run_large, method="serial", method_kwargs={"callbacks": [TqdmCallback(max_evals)]}
)
search_large = CBO(problem_large, evaluator_large, random_state=42)
results["Large"] = search_large.search(max_evals)

# %%
evaluator_large_tl = Evaluator.create(
    run_large, method="serial", method_kwargs={"callbacks": [TqdmCallback(max_evals)]}
)
search_large_tl = CBO(problem_large, evaluator_large_tl, random_state=42)
search_large_tl.fit_generative_model(results["Large"])
results["Large+TL"] = search_large_tl.search(max_evals)

# %%
# Finally, we compare the results and quickly see that transfer-learning provided a consequant speed-up for the search:
import matplotlib.pyplot as plt

plt.figure()

for strategy, df in results.items():
    x = [i for i in range(len(df))]
    plt.scatter(x, df.objective, label=strategy)
    plt.plot(x, df.objective.cummax())

plt.xlabel("Time (sec.)")
plt.ylabel("Objective")
plt.grid()
plt.legend()
plt.show()

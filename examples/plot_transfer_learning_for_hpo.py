# -*- coding: utf-8 -*-
"""
Applying Transfer Learning to Hyperparameter Optimization
=========================================================

**Author(s)**: Romain Egele.

In this example, we demonstrate how to leverage transfer learning for hyperparameter optimization. Imagine you are working on multiple related tasks, such as optimizing the hyperparameters of neural networks for various datasets. It's reasonable to expect that similar hyperparameter configurations might perform well across these datasets, even if some minor adjustments are needed to fine-tune performance.

By conducting a thorough (and potentially expensive) search on one task, you can reuse the resulting hyperparameter set to guide and accelerate optimization for subsequent tasks. This approach reduces computational costs while maintaining high performance.

To illustrate, we will use a simple and computationally inexpensive example: minimizing the function :math:`f(x) = \\sum_{i=0}^
{n-1}`. Here, the difficulty of the problem is defined by the number of variables :math:`n`. We'll start by optimizing the small problem where :math:`n=1`. Then, we’ll apply transfer learning to optimize a larger problem where :math:`n=3`, comparing the results with and without transfer learning to highlight the benefits.

Let's begin by defining the run-functions for both the small-scale and large-scale problems:
"""

# %%
import functools

import matplotlib.pyplot as plt

from deephyper.analysis import figure_size
from deephyper.analysis.hpo import plot_search_trajectory_single_objective_hpo
from deephyper.evaluator import Evaluator
from deephyper.evaluator.callback import TqdmCallback
from deephyper.hpo import CBO, HpProblem


def run(config: dict, N: int) -> float:
    # Definition of the function to minimize
    y = sum([config[f"x{i}"] ** 2 for i in range(N)])
    return -y  # Use the `-` sign to perform minimization


n_small = 1
n_large = 3
run_small = functools.partial(run, N=n_small)
run_large = functools.partial(run, N=n_large)

# %%
# Then, we can define the hyperparameter problem space based on :math:`n`

N = n_small
problem_small = HpProblem()
for i in range(N):
    problem_small.add_hyperparameter((-10.0, 10.0), f"x{i}")
problem_small

# %%

N = n_large
problem_large = HpProblem()
for i in range(N):
    problem_large.add_hyperparameter((-10.0, 10.0), f"x{i}")
problem_large

# %%
# Then, we define setup the search and execute it:

results = {}
max_evals = 100
evaluator_small = Evaluator.create(
    run_small, method="thread", method_kwargs={"callbacks": [TqdmCallback()]}
)
search_small = CBO(problem_small, evaluator_small, random_state=42)
results_small = search_small.search(max_evals)

# %%

evaluator_large = Evaluator.create(
    run_large, method="thread", method_kwargs={"callbacks": [TqdmCallback()]}
)
search_large = CBO(problem_large, evaluator_large, random_state=42)
results["Large"] = search_large.search(max_evals)

# %%

evaluator_large_tl = Evaluator.create(
    run_large, method="thread", method_kwargs={"callbacks": [TqdmCallback()]}
)
search_large_tl = CBO(problem_large, evaluator_large_tl, random_state=42)
search_large_tl.fit_generative_model(results_small)
results["Large+TL"] = search_large_tl.search(max_evals)

# %%
# Finally, we compare the results and quickly see that transfer-learning
# provided a consequant speed-up for the search:

fig, ax = plt.subplots(figsize=figure_size(width=600))

for strategy, df in results.items():
    plot_search_trajectory_single_objective_hpo(
        df,
        show_failures=False,
        mode="min",
        ax=ax,
        label=strategy,
    )

plt.xlabel("Time (sec.)")
plt.ylabel("Objective")
plt.yscale("log")
plt.grid(visible=True, which="minor", linestyle=":")
plt.grid(visible=True, which="major", linestyle="-")
plt.legend()
plt.show()

# %%

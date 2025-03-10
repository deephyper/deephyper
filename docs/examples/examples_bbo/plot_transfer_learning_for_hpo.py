r"""
Applying Transfer Learning to Black-Box Optimization
====================================================

**Author(s)**: Romain Egele.

In this example, we demonstrate how to leverage transfer learning for hyperparameter optimization. Imagine you are working on multiple related tasks, such as optimizing the hyperparameters of neural networks for various datasets. It's reasonable to expect that similar hyperparameter configurations might perform well across these datasets, even if some minor adjustments are needed to fine-tune performance.

By conducting a thorough (and potentially expensive) search on one task, you can reuse the resulting hyperparameter set to guide and accelerate optimization for subsequent tasks. This approach reduces computational costs while maintaining high performance.

To illustrate, we will use a simple and computationally inexpensive example: minimizing the function :math:`f(x) = \sum_{i=0}^
{n-1}`. Here, the difficulty of the problem is defined by the number of variables :math:`n`. We will start by optimizing the small problem where :math:`n=5`. Then, we will apply transfer learning to optimize a larger problem where :math:`n=10`, comparing the results with and without transfer learning to highlight the benefits.

Let's begin by defining the run-functions for both the small-scale and large-scale problems:
"""

# %%

# .. dropdown:: Import statements
import functools

import matplotlib.pyplot as plt

from deephyper.analysis import figure_size
from deephyper.analysis.hpo import plot_search_trajectory_single_objective_hpo
from deephyper.evaluator import Evaluator
from deephyper.evaluator.callback import TqdmCallback
from deephyper.hpo import CBO, HpProblem

WIDTH_PLOTS = 8
HEIGHT_PLOTS = WIDTH_PLOTS / 1.618

# %%
def run(job, N: int) -> float:
    # Definition of the function to minimize
    y = sum([job.parameters[f"x{i}"] ** 2 for i in range(N)])
    return -y  # Use the `-` sign to perform minimization


n_small = 5
n_large = 10
run_small = functools.partial(run, N=n_small)
run_large = functools.partial(run, N=n_large)

# %%
# Then, we can define the hyperparameter problem space based on :math:`n`

def create_problem(n):
    problem = HpProblem()
    for i in range(n):
        problem.add_hyperparameter((-10.0, 10.0), f"x{i}")
    return problem

# %%
problem_small = create_problem(n_small)

# %%
problem_large = create_problem(n_large)

# %%
# We define the parameters of the search:
search_kwargs = {
    "surrogate_model": "ET",  # Use Extra Trees as surrogate model
    "surrogate_model_kwargs": {
        "n_estimators": 25,  # Relatively small number of trees in the surrogate to make it "fast"
        "min_samples_split": 4,  # Larger number to avoid small leaf nodes (smoothing the objective response)
    },
    "acq_optimizer": "ga", # Optimizing the acquisition function with countinuous genetic algorithm
    "acq_optimizer_freq": 1,
    "filter_duplicated": False,  # Deactivate filtration of duplicated new points
    "kappa": 10.0,  # Initial value of exploration-exploitation parameter for the acquisition function
    "scheduler": {  # Scheduler for the exploration-exploitation parameter "kappa"
        "type": "periodic-exp-decay",  # Periodic exponential decay
        "period": 25,  # Period over which the decay is applied. It is useful to escape local solutions.
        "kappa_final": 0.01,  # Value of kappa at the end of each "period"
    },
    "objective_scaler": "identity",
    "random_state": 42,
}

# %%
# We create a dictionnary that will store the results of each experiment and also fix the number of
# evaluation of the search to 200.
results = {}
max_evals = 200

# %%
# Then, we run the search for each problem. We start with the small problem:
evaluator_small = Evaluator.create(
    run_small, 
    method="thread", 
    method_kwargs={"callbacks": [TqdmCallback("HPO - Small Problem")]},
)

search_small = CBO(
    problem_small, 
    evaluator_small, 
    n_initial_points=2 * n_small + 1, 
    **search_kwargs,
)
results_small = search_small.search(max_evals)

# %%
# We run the search on the large problem without transfer learning:
evaluator_large = Evaluator.create(
    run_large,
    method="thread",
    method_kwargs={"callbacks": [TqdmCallback("HPO - Large Problem")]},
)
search_large = CBO(
    problem_large, 
    evaluator_large,
    n_initial_points=2 * n_large + 1,
    **search_kwargs,
)
results["Large"] = search_large.search(max_evals)

# %%
# Finally, we run the search on the large problem with transfer learning from the results
# of the small problem that we computed first:
evaluator_large_tl = Evaluator.create(
    run_large,
    method="thread",
    method_kwargs={"callbacks": [TqdmCallback("HPO - Large Problem with TL")]},
)
search_large_tl = CBO(
    problem_large, 
    evaluator_large_tl, 
    n_initial_points=2 * n_large + 1, 
    **search_kwargs,
)

# This is where transfer learning happens
search_large_tl.fit_generative_model(results_small)

results["Large+TL"] = search_large_tl.search(max_evals)


# %%
# Finally, we compare the results and quickly see that transfer-learning
# provided a consequant speed-up for the search:
fig, ax = plt.subplots(figsize=(WIDTH_PLOTS, HEIGHT_PLOTS), tight_layout=True)

for i, (strategy, df) in enumerate(results.items()):
    plot_search_trajectory_single_objective_hpo(
        df,
        show_failures=False,
        mode="min",
        ax=ax,
        label=strategy,
        plot_kwargs={"color": f"C{i}"},
        scatter_success_kwargs={"c": f"C{i}"},
    )

ax.set_xlabel
ax.set_xlabel("Time (sec.)")
ax.set_ylabel("Objective")
ax.set_yscale("log")
ax.grid(visible=True, which="minor", linestyle=":")
ax.grid(visible=True, which="major", linestyle="-")
ax.legend()
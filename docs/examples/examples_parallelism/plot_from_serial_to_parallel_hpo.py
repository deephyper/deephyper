# -*- coding: utf-8 -*-
"""
From Sequential to Massively-Parallel Bayesian Optimization
===========================================================

**Author(s)**: Romain Egele.

In this example you will learn about the advantages of parallel over sequential
evaluations with Bayesian optimization. 

We start by defining a black-box ``run``-function that implements the Ackley function:

.. image:: https://www.sfu.ca/~ssurjano/ackley.png
  :width: 400
  :alt: Ackley Function in 2D
"""
# %%

# .. dropdown:: Import statements
import time

import matplotlib.pyplot as plt
import numpy as np

from deephyper.analysis import figure_size
from deephyper.analysis.hpo import plot_search_trajectory_single_objective_hpo
from deephyper.analysis.hpo import plot_worker_utilization
from deephyper.evaluator import Evaluator, profile
from deephyper.evaluator.callback import TqdmCallback
from deephyper.hpo import HpProblem, CBO, RandomSearch

# %%
# We define the Ackley function:

# .. dropdown:: Ackley function
def ackley(x, a=20, b=0.2, c=2 * np.pi):
    d = len(x)
    s1 = np.sum(x**2)
    s2 = np.sum(np.cos(c * x))
    term1 = -a * np.exp(-b * np.sqrt(s1 / d))
    term2 = -np.exp(s2 / d)
    y = term1 + term2 + a + np.exp(1)
    return y

# %% 
# We will use the ``time.sleep`` function to simulate a budget of 2 secondes of execution in average 
# which helps illustrate the advantage of parallel evaluations. The ``@profile`` decorator is useful 
# to collect starting/ending time of the ``run``-function execution which help us know exactly when 
# we are inside the black-box. This decorator is necessary when profiling the worker utilization. When 
# using this decorator, the ``run``-function will return a dictionnary with 2 new keys ``"timestamp_start"`` 
# and ``"timestamp_end"``.

@profile
def run_ackley(config, sleep_loc=2, sleep_scale=0.5):
    # to simulate the computation of an expensive black-box
    if sleep_loc > 0:
        t_sleep = np.random.normal(loc=sleep_loc, scale=sleep_scale)
        t_sleep = max(t_sleep, 0)
        time.sleep(t_sleep)

    x = np.array([config[k] for k in config if "x" in k])
    x = np.asarray_chkfinite(x)  # ValueError if any NaN or Inf
    return -ackley(x)  # maximisation is performed

# %%
# Then, we define the variable(s) we want to optimize. For this problem we
# optimize Ackley in a N-dimensional search space. Each dimension in the continuous range
# [-32.768, 32.768]. The true minimum is located at ``(0, ..., 0)``.

nb_dim = 5
problem = HpProblem()
for i in range(nb_dim):
    problem.add_hyperparameter((-32.768, 32.768), f"x{i}")
problem

# %%
# Then, we define some default search parameters for the Centralized Bayesian Optimization (CBO) algorithm.
search_kwargs = {
    "multi_point_strategy": "qUCBd", # Multi-point strategy for asynchronous batch generations (explained later)
    "acq_optimizer": "ga", # Use continuous Genetic Algorithm for the acquisition function optimizer
    "filter_duplicated": False, # Deactivate filtration of duplicated new points
    "random_state": 42, # Random seed
}

# %%
# Then, we define the time budget for the optimization. We will compare the performance of a sequential
# search with a parallel search for the same time budget. The time budget is defined in seconds.
timeout = 60 # 1 minute

# %%
# Then, we define the sequential Bayesian optimization search.
sequential_search = CBO(problem, run_ackley, **search_kwargs)

# %%
# The previously simplified definition of the search is equivalent to the following:
sequential_evaluator = Evaluator.create(
    run_ackley,
    method="thread",  # For synchronous function defintion relying on the GIL or I/O bound tasks
    method_kwargs={
        "num_workers": 1, 
        "callbacks": [TqdmCallback("Sequential BO:")]
    },
)
sequential_search = CBO(problem, sequential_evaluator, **search_kwargs)

# %%
# We define a utility function to preprocess our results before plotting.

def preprocess_results(results):
    results = results.dropna().copy()
    offset = results["m:timestamp_start"].min()
    results["m:timestamp_start"] -= offset
    results["m:timestamp_end"] -= offset
    return results

# %%
# Where we use the ``"thread"``-evaluator with a single worker and use the ``TqdmCallback`` to display
# a progress bar during the search. 
# 
# We can now run the sequential search for 2 minutes. The call to the ``search``-method will return a
# DataFrame with the results of the search.
#
# If this step is executed multiple times without creating a new search the results will be accumulated in the same DataFrame.

results = {}
results["sequential"] = preprocess_results(sequential_search.search(timeout=timeout))
results["sequential"]

# %%
# Each row of the DataFrame corresponds to an evaluation of the ``run``-function. The DataFrame contains the following columns:
# - ``"p:*"``: The parameters of the search space.
# - ``"objective"``: The objective value returned by the evaluation.
# - ``"job_id"``: The id of the evaluation in increasing order of job creation.
# - ``"job_status"``: The status of the evaluation (e.g., "DONE", "CANCELLED").
# - ``"m:timestamp_submit/gather"``: The submition and gathering times of the evaluation by the ``Evaluator`` (includes overheads).
# - ``"m:timestamp_start/end"``: The starting and ending time of the evaluation.
#
# We can now plot the results of the sequential search. The first plot shows the evolution of the objective.
# The second plot shows the utilization of the worker over time.

# .. dropdown:: Plot search trajectory and worker utilization
fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        sharex=True,
        figsize=figure_size(width=600),
        tight_layout=True,
    )

_ = plot_search_trajectory_single_objective_hpo(
    results["sequential"], mode="min", x_units="seconds", ax=axes[0]
)

_ = plot_worker_utilization(
    results["sequential"], num_workers=1, profile_type="start/end", ax=axes[1]
)

# %%
# Then, we can create a parallel evaluator with 100 workers.
parallel_evaluator = Evaluator.create(
    run_ackley,
    method="thread",
    method_kwargs={
        "num_workers": 100, # For the parallel evaluations
        "callbacks": [TqdmCallback("Parallel BO:")]
    },
)
parallel_search = CBO(problem, parallel_evaluator, **search_kwargs)

# %%
# The parallel search is executed for 1 minute.
results["parallel"] = preprocess_results(parallel_search.search(timeout=timeout))
results["parallel"]

# %%
# It can be surprising to see in the results that the last lines have ``"job_status"`` set to "CANCELLED" but
# still have an objective value. This is due to the fact that the cancellation of a job is asynchronous and already scheduled Asyncio tasks are therefore executed. When the timeout is reached the jobs created by the "thread" method jobs cannot be directly killed but rather their ``job.status`` is updated to ``"CANCELLING"`` and the user-code is responsible for checking the status of the job and interrupting the execution. This is why the objective value is still present in the results. This behavior is different from the "process" method where the jobs are killed directly.
#
# We can now plot the results of the parallel search. The first plot shows the evolution of the objective.
# The second plot shows the utilization of the worker over time.
#
# We can see that the parallel search is able to evaluate a lot more points in the same time budget. This also
# allows the algorithm to explore more of the search space and potentially find better solutions.
# The utilization plot shows that the workers are used efficiently in the parallel search (above 80%).

# .. dropdown:: Plot search trajectory and worker utilization
fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        sharex=True,
        figsize=figure_size(width=600),
        tight_layout=True,
    )

_ = plot_search_trajectory_single_objective_hpo(
    results["parallel"], mode="min", x_units="seconds", ax=axes[0]
)

_ = plot_worker_utilization(
    results["parallel"], num_workers=1, profile_type="start/end", ax=axes[1]
)

# %%
# Finally, we compare both search with the execution time is used as the x-axis.
# The advantage of parallelism is clearly visible by the difference in the number of evaluations and in objective.

# .. dropdown:: Plot search trajectories
fig, ax = plt.subplots(figsize=figure_size(width=600), tight_layout=True)

for i, (strategy, df) in enumerate(results.items()):
    plot_search_trajectory_single_objective_hpo(
        df,
        show_failures=False,
        mode="min",
        x_units="seconds",
        ax=ax,
        label=strategy,
        plot_kwargs={"color": f"C{i}"},
        scatter_success_kwargs={"color": f"C{i}"},
    )

_ = ax.set_xlabel("Time (sec.)")
_ = ax.set_ylabel("Objective")
_ = ax.set_yscale("log")
_ = ax.grid(visible=True, which="minor", linestyle=":")
_ = ax.grid(visible=True, which="major", linestyle="-")
_ = ax.legend()



# %%
# Finally, one could compare to a random search to see if the overheads of the parallel Bayesian optimization are worth it (i.e., the cost of fitting and optimizing the surrogate model).
# The evaluator is defined similarly to the one used for the parallel Bayesian optimization search:

parallel_evaluator = Evaluator.create(
    run_ackley,
    method="thread",
    method_kwargs={
        "num_workers": 100, # For the parallel evaluations
        "callbacks": [TqdmCallback("Random Search:")]
    },
)
random_search = RandomSearch(problem, parallel_evaluator, random_state=search_kwargs["random_state"])
results["random"] = preprocess_results(random_search.search(timeout=timeout))
results["random"]


# %%
# The number of evaluations of the random search is higher than the parallel Bayesian optimization search.
print(f"Number of evaluations for the parallel Bayesian optimization: {len(results['parallel'])}")
print(f"Number of evaluations for the random search: {len(results['random'])}")

# %%
# The utilization of the worker is confirmed to be near 100% for the random search.

# .. dropdown:: Plot search trajectory and worker utilization
fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        sharex=True,
        figsize=figure_size(width=600),
        tight_layout=True
    )

_ = plot_search_trajectory_single_objective_hpo(
    results["random"], mode="min", x_units="seconds", ax=axes[0]
)

_ = plot_worker_utilization(
    results["random"], num_workers=1, profile_type="start/end", ax=axes[1]
)

# %%
# However, the objective value of the parallel Bayesian optimization search is significantly better than the random search.

# .. dropdown:: Plot search trajectories
# sphinx_gallery_thumbnail_number = 5
fig, ax = plt.subplots(figsize=figure_size(width=600), tight_layout=True)
labels = {
    "random": "Parallel Random Search",
    "sequential": "Sequential Bayesian Optimization",
    "parallel": "Parallel Bayesian Optimization",
    }
for i, (key, label) in enumerate(labels.items()):
    df = results[key]
    _ = plot_search_trajectory_single_objective_hpo(
        df,
        show_failures=False,
        mode="min",
        x_units="seconds",
        ax=ax,
        label=label,
        plot_kwargs={"color": f"C{i}"},
        scatter_success_kwargs={"color": f"C{i}", "alpha": 0.5},
    )

_ = ax.set_xlabel("Time (sec.)")
_ = ax.set_ylabel("Objective")
_ = ax.set_yscale("log")
_ = ax.grid(visible=True, which="minor", linestyle=":")
_ = ax.grid(visible=True, which="major", linestyle="-")
_ = ax.legend()

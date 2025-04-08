r"""
Scaling Bayesian Optimization with Heterogeneous Parallelism
============================================================

**Author(s)**: Romain Egele.

This example demonstrates the advantages of mixing centralized parallelism (1 optimization process with N workers)
with decentralized parallelism (N optimization processes) to scale Bayesian optimization. For this we will have
a total of 1000 local workers simulated with threads and timeouts.

In this example, we will start by demonstrating the behaviour of an efficient centralized bayesian optimization using 1000 workers.
Then, we will run a mixed decentralized optimization with 10 replications of a centralized optimization each with 100 workers for a
total of 1000 workers as well.

Therefore, we start by defining a black-box ``run``-function that implements the Ackley function:

.. image:: https://www.sfu.ca/~ssurjano/ackley.png
  :width: 400
  :alt: Ackley Function in 2D
"""
# %%

# .. dropdown:: Import statements
import multiprocessing
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss

from loky import get_reusable_executor

from deephyper.analysis import figure_size
from deephyper.analysis.hpo import plot_search_trajectory_single_objective_hpo
from deephyper.analysis.hpo import plot_worker_utilization
from deephyper.evaluator import Evaluator, profile
from deephyper.evaluator.callback import TqdmCallback
from deephyper.evaluator.storage import SharedMemoryStorage
from deephyper.hpo import HpProblem, CBO

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

nb_dim = 10
problem = HpProblem()
for i in range(nb_dim):
    problem.add_hyperparameter((-32.768, 32.768), f"x{i}")
problem

# %%
# Then, we define some default search parameters for the Bayesian Optimization algorithm.
# It is important to note the `"qUCBd"` parameter for the multi-point strategy. Using the
# classic constant-liar strategy (a.k.a, Krigging Believer) `"cl_min/max/mean` in our setting
# would totally freeze the execution (you can try!).
search_kwargs = {
    "acq_func_kwargs": {
        "kappa": 2.0,
    },
    "acq_optimizer": "ga",  # Use continuous Genetic Algorithm for the acquisition function optimizer
    "acq_optimizer_kwargs": {
        "filter_duplicated": False, # Deactivate filtration of duplicated new points
    },
    "multi_point_strategy": "qUCBd",  # Multi-point strategy for asynchronous batch generations (explained later)
    "random_state": 42,  # Random seed
}

# %%
# Then, we define the time budget for the optimization. The time budget is defined in seconds.
# The `sleep_loc` and `sleep_scale` parameters simulate the distribution of duration of evaluated
# black-box functions sampled from a normal law with mean `sleep_loc` and standard deviation `sleep_scale`.
# We also define here the total number of workers to 1000.
# Using so many workers for Bayesian optimization is quite rare. Usually it is limited to ~200 sequential
# iterations and a dozen of parallel workers.
timeout = 30
num_workers = 1_000
run_function_kwargs = dict(sleep_loc=1, sleep_scale=0.25)
results = {}


# %%
# We define a utility function to preprocess our results before plotting.
def preprocess_results(results):
    results = results.dropna().copy()
    offset = results["m:timestamp_start"].min()
    results["m:timestamp_start"] -= offset
    results["m:timestamp_end"] -= offset
    return results


# %%
# Then, we can create a centralized parallel search with .
def execute_centralized_bo(
    problem, run_function, run_function_kwargs, num_workers, log_dir, search_kwargs, timeout
):
    evaluator = Evaluator.create(
        run_function,
        method="thread",
        method_kwargs={
            "num_workers": num_workers,  # For the parallel evaluations
            "callbacks": [TqdmCallback()],
            "run_function_kwargs": run_function_kwargs,
        },
    )
    search = CBO(
        problem,
        evaluator,
        log_dir=log_dir,
        **search_kwargs,
    )
    results = search.search(timeout=timeout)

    results = preprocess_results(results)

    return results


# %%
# To execute the search, we use the ``if __name__ == "__main__":`` statement. It is important to avoid triggering searches
# recursively when launching child processes latter in the example.
if __name__ == "__main__":
    results["centralized"] = execute_centralized_bo(
        problem=problem,
        run_function=run_ackley,
        run_function_kwargs=run_function_kwargs,
        num_workers=num_workers,
        log_dir="search_centralized",
        search_kwargs=search_kwargs,
        timeout=timeout,
    )

# %%
# We can now plot the results of the centralized search. The first plot shows the evolution of the objective.
# The second plot shows the utilization of the worker over time.
# The drops of the function show the regular re-fit of the surrogate model and optimization of the acquisition function.

# .. dropdown:: Make plot
if __name__ == "__main__":
    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        sharex=True,
        figsize=figure_size(width=600),
        tight_layout=True,
    )

    fig, ax = plot_search_trajectory_single_objective_hpo(
        results["centralized"], mode="min", x_units="seconds", ax=axes[0]
    )

    fig, ax = plot_worker_utilization(
        results["centralized"], num_workers=None, profile_type="start/end", ax=axes[1]
    )


# %%
# Then we move to the decentralized optimization. We defined it in a separate module for Pickling to work.
# This function will be launched in child processes to trigger each sub-instances of the decentralized search.

def execute_centralized_bo_with_share_memory(
    problem,
    run_function,
    run_function_kwargs,
    storage,
    search_id,
    search_random_state,
    log_dir,
    num_workers,
    is_master,
    kappa,
    search_kwargs,
    timeout,
):
    evaluator = Evaluator.create(
        run_function,
        method="thread",
        method_kwargs={
            "num_workers": num_workers,
            "storage": storage,
            "search_id": search_id,
            "callbacks": [TqdmCallback()] if is_master else [],
            "run_function_kwargs": run_function_kwargs,
        },
    )

    search_kwargs["acq_func_kwargs"]["kappa"] = kappa
    search_kwargs["random_state"] = search_random_state
    search = CBO(problem, evaluator, log_dir=log_dir, **search_kwargs)

    def dummy(*args, **kwargs):
        pass

    results = None
    if is_master:
        results = search.search(timeout=timeout)
    else:
        # for concurrency reasons this is important to override these functions
        evaluator.dump_jobs_done_to_csv = dummy
        search.extend_results_with_pareto_efficient = dummy

        search.search(timeout=timeout)

    return results

# %%
# This function is very similar to our previous centralized optimization. However, importantly you can see that we
# override two methods of the ``CBO`` with ``dummy`` function.
#
# Now we can define the decentralized search by launching centralized instances with an paralle executor.
# Importantly we use ``SharedMemoryStorage`` so that centralized sub-instances can communicate globally.
# We also create explicitely the ``search_id`` to make sure they communicate about the search Search instance.
# The ``kappa`` value is sampled from an Exponential distribution to enable a diverse set of exploration-exploitation trade-offs
# leading to better results. You can try to fix it to ``1.96`` instead (default parameter in DeepHyper).
def execute_decentralized_bo(
    problem,
    run_function,
    run_function_kwargs,
    num_workers,
    log_dir,
    search_kwargs,
    timeout,
    n_processes,
):
    storage = SharedMemoryStorage()
    search_id = storage.create_new_search()
    kappa = ss.expon.rvs(
        size=n_processes, 
        scale=search_kwargs["acq_func_kwargs"]["kappa"], 
        random_state=search_kwargs["random_state"],
    )

    executor = get_reusable_executor(max_workers=n_processes, context="spawn")
    futures = []
    for i in range(n_processes):
        future = executor.submit(execute_centralized_bo_with_share_memory, *(
            problem,
            run_function,
            run_function_kwargs,
            storage,
            search_id,
            i,
            log_dir,
            num_workers // n_processes,
            i == 0,
            kappa[i],
            search_kwargs,
            timeout
            )
        )
        futures.append(future)

    results = preprocess_results(futures[0].result())

    return results


# %%
# We can now execute the decentralized optimization with 10 processes each using 100 workers:
if __name__ == "__main__":
    results["decentralized"] = execute_decentralized_bo(
        problem=problem,
        run_function=run_ackley,
        run_function_kwargs=run_function_kwargs,
        num_workers=num_workers,
        log_dir="search_decentralized",
        search_kwargs=search_kwargs,
        timeout=timeout,
        n_processes=multiprocessing.cpu_count(),
    )

# %%
# Observing the results we can see a better objective and less intance drops in worker utilization:

# .. dropdown:: Make plot
if __name__ == "__main__":
    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        sharex=True,
        figsize=figure_size(width=600),
        tight_layout=True,
    )

    fig, ax = plot_search_trajectory_single_objective_hpo(
        results["decentralized"], mode="min", x_units="seconds", ax=axes[0]
    )

    fig, ax = plot_worker_utilization(
        results["decentralized"], num_workers=None, profile_type="start/end", ax=axes[1]
    )

# %%
# If we compare the objective curves side by side we can see the improvement of decentralized
# optimization even better.
# Even if the total number of evaluations is less, the quality is much better as we infered more
# often the surrogate model (combining re-fitting and optimization of the acquisition function):

# .. dropdown:: Make plot
if __name__ == "__main__":
    # sphinx_gallery_thumbnail_number = 3
    fig, ax = plt.subplots(figsize=figure_size(width=600), tight_layout=True)
    labels = {
        "centralized": "Centralized Bayesian Optimization",
        "decentralized": "Decentralized Bayesian Optimization",
    }

    x_min = float("inf")
    x_max = -float("inf")
    for i, (key, label) in enumerate(labels.items()):
        df = results[key]
        plot_search_trajectory_single_objective_hpo(
            df,
            show_failures=False,
            mode="min",
            x_units="seconds",
            ax=ax,
            label=label,
            plot_kwargs={"color": f"C{i}"},
            scatter_success_kwargs={"color": f"C{i}", "alpha": 0.5},
        )
        x_min = min(df["m:timestamp_start"].min(), x_min)
        x_max = max(df["m:timestamp_end"].max(), x_max)

    ax.set_xlim(x_min, x_max)

    ax.set_xlabel("Time (sec.)")
    ax.set_ylabel("Objective")
    ax.set_yscale("log")
    ax.grid(visible=True, which="minor", linestyle=":")
    ax.grid(visible=True, which="major", linestyle="-")
    ax.legend()

# -*- coding: utf-8 -*-
"""
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

To help illustrate the parallelization gain, we will simulate a computational cost
by using ``time.sleep``. We also use the ``@profile`` decorator to collect starting/ending
times of each call to the ``run``-function. When using this decorator, the ``run``-function will
return a dictionnary including ``"metadata"`` with 2 new keys ``"timestamp_start"`` and
``"timestamp_end"``. The ``run``-function is defined in a separate Python module
for better serialization (through ``pickle``) in case other parallel backends such as ``"process"`` would be used

.. literalinclude:: ../../examples/black_box_util.py
   :language: python

After defining the ``run``-function we can continue with the definition of our optimization script:
"""

# %%
# A few useful imports
import black_box_util as black_box
import matplotlib.pyplot as plt
import scipy.stats as ss

from multiprocessing import Pool

from deephyper.analysis import figure_size
from deephyper.analysis.hpo import plot_search_trajectory_single_objective_hpo
from deephyper.analysis.hpo import plot_worker_utilization
from deephyper.evaluator import Evaluator
from deephyper.evaluator.callback import TqdmCallback
from deephyper.evaluator.storage import SharedMemoryStorage
from deephyper.hpo import HpProblem, CBO

from dbo_util import execute_centralized_bo_with_share_memory

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
# Then, we define some default search parameters for the Bayesian Optimization algorithm.
# It is important to note the `"qUCBd"` parameter for the multi-point strategy. Using the
# classic constant-liar strategy (a.k.a, Krigging Believer) `"cl_min/max/mean` in our setting
# would totally freeze the execution (you can try!).
search_kwargs = {
    "n_initial_points": 2 * nb_dim + 1,  # Number of initial random points
    "surrogate_model": "ET",  # Use Extra Trees as surrogate model
    "surrogate_model_kwargs": {
        "n_estimators": 25,  # Relatively small number of trees in the surrogate to make it "fast"
        "min_samples_split": 8,  # Larger number to avoid small leaf nodes (smoothing the response)
    },
    "multi_point_strategy": "qUCBd",  # Multi-point strategy for asynchronous batch generations (explained later)
    "acq_optimizer": "ga",  # Use continuous Genetic Algorithm for the acquisition function optimizer
    "acq_optimizer_freq": 1,  # Frequency of the acquisition function optimizer (1 = each new batch generation) increasing this value can help amortize the computational cost of acquisition function optimization
    "filter_duplicated": False,  # Deactivate filtration of duplicated new points
    "kappa": 10.0,  # Initial value of exploration-exploitation parameter for the acquisition function
    "scheduler": {  # Scheduler for the exploration-exploitation parameter "kappa"
        "type": "periodic-exp-decay",  # Periodic exponential decay
        "period": 50,  # Period over which the decay is applied. It is useful to escape local solutions.
        "kappa_final": 0.001,  # Value of kappa at the end of each "period"
    },
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
# To execute the search, we use the `if __name__ == "__main__":` statement. It is important to avoid triggering searches
# recursively when launching child processes latter in the example.
if __name__ == "__main__":
    results["centralized"] = execute_centralized_bo(
        problem=problem,
        run_function=black_box.run_ackley,
        run_function_kwargs=run_function_kwargs,
        num_workers=num_workers,
        log_dir="search_centralized",
        search_kwargs=search_kwargs,
        timeout=timeout,
    )
    results["centralized"]

# %%
# We can now plot the results of the centralized search. The first plot shows the evolution of the objective.
# The second plot shows the utilization of the worker over time.
# The drops of the function show the regular re-fit of the surrogate model and optimization of the acquisition function.
if __name__ == "__main__":
    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        sharex=True,
        figsize=figure_size(width=600),
    )

    plot_search_trajectory_single_objective_hpo(
        results["centralized"], mode="min", x_units="seconds", ax=axes[0]
    )

    plot_worker_utilization(
        results["centralized"], num_workers=None, profile_type="start/end", ax=axes[1]
    )

    plt.tight_layout()
    plt.show()


# %%
# Then we move to the decentralized optimization. We defined it in a separate module for Pickling to work.
# This function will be launched in child processes to trigger each sub-instances of the decentralized search.
#
# .. literalinclude:: ../../examples/dbo_util.py
#    :language: python
#
# This function is very similar to our previous centralized optimization. However, importantly you can see that we
# override two methods of the `CBO` with `dummy` function.
#
# Now we can define the decentralized search using that launches each centralized instance with `multiprocessing.Pool`.
# Importantly we use `SharedMemoryStorage` so that centralized sub-instances can communicate globally.
# We also create explicitely the `search_id` to make sure they communicate about the search Search instance.
# The `kappa` value is sampled from an Exponential distribution to enable a diverse set of exploration-exploitation trade-offs
# leading to better results. You can try to fix it to `1.96` instead (default parameter in DeepHyper).
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
        size=n_processes, scale=search_kwargs["kappa"], random_state=search_kwargs["random_state"]
    )
    with Pool(processes=n_processes) as pool:
        results = pool.starmap(
            execute_centralized_bo_with_share_memory,
            [
                (
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
                    timeout,
                )
                for i in range(n_processes)
            ],
        )

    results = preprocess_results(results[0])

    return results


# %%
# We can now execute the decentralized optimization with 10 processes each using 100 workers:
if __name__ == "__main__":
    results["decentralized"] = execute_decentralized_bo(
        problem=problem,
        run_function=black_box.run_ackley,
        run_function_kwargs=run_function_kwargs,
        num_workers=num_workers,
        log_dir="search_centralized",
        search_kwargs=search_kwargs,
        timeout=timeout,
        n_processes=10,
    )
    results["decentralized"]

# %%
# Observing the results we can see a better objective and less intance drops in worker utilization:
if __name__ == "__main__":
    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        sharex=True,
        figsize=figure_size(width=600),
    )

    plot_search_trajectory_single_objective_hpo(
        results["decentralized"], mode="min", x_units="seconds", ax=axes[0]
    )

    plot_worker_utilization(
        results["decentralized"], num_workers=None, profile_type="start/end", ax=axes[1]
    )

    plt.tight_layout()
    plt.show()

# %%
# If we compare the objective curves side by side we can see the improvement of decentralized
# optimization even better.
# Even if the total number of evaluations is less, the quality is much better as we infered more
# often the surrogate model (combining re-fitting and optimization of the acquisition function):
if __name__ == "__main__":
    # sphinx_gallery_thumbnail_number = 3
    fig, ax = plt.subplots(figsize=figure_size(width=600))
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

    plt.xlabel("Time (sec.)")
    plt.ylabel("Objective")
    plt.yscale("log")
    plt.grid(visible=True, which="minor", linestyle=":")
    plt.grid(visible=True, which="major", linestyle="-")
    plt.legend()
    plt.show()

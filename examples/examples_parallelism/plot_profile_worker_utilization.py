r"""
Profile the Worker Utilization
==============================

**Author(s)**: Romain Egele.

In this example, you will learn how to profile the activity of workers during a 
search. 

We start by defining an artificial black-box ``run``-function by using the Ackley function:

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
from deephyper.analysis.hpo import (
    plot_search_trajectory_single_objective_hpo,
    plot_worker_utilization,
)
from deephyper.evaluator import Evaluator, profile
from deephyper.evaluator.callback import TqdmCallback
from deephyper.hpo import CBO, HpProblem

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
# Then we define the variable(s) we want to optimize. For this problem we
# optimize Ackley in a 2-dimensional search space, the true minimul is
# located at ``(0, 0)``.

def create_problem(nb_dim=2):
    nb_dim = 2
    problem = HpProblem()
    for i in range(nb_dim):
        problem.add_hyperparameter((-32.768, 32.768), f"x{i}")
    return problem

problem = create_problem()
problem


# %%
# Then we define a parallel search.
#  As the ``run``-function is defined in the same module  we use the "loky" backend 
# that serialize by value.
def execute_search(timeout, num_workers):

    evaluator = Evaluator.create(
        run_ackley,
        method="loky",
        method_kwargs={
            "num_workers": num_workers,
            "callbacks": [TqdmCallback()],
        },
    )

    search = CBO(
        problem,
        evaluator,
        multi_point_strategy="qUCBd",
        random_state=42,
    )

    results = search.search(timeout=timeout)

    return results

if __name__ == "__main__":
    timeout = 20
    num_workers = 4
    results = execute_search(timeout, num_workers)

# %%
# Finally, we plot the results from the collected DataFrame.

# .. dropdown:: Plot search trajectory an workers utilization
if __name__ == "__main__":
    t0 = results["m:timestamp_start"].iloc[0]
    results["m:timestamp_start"] = results["m:timestamp_start"] - t0
    results["m:timestamp_end"] = results["m:timestamp_end"] - t0
    tmax = results["m:timestamp_end"].max()

    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        sharex=True,
        figsize=figure_size(width=600),
        tight_layout=True,
    )

    _ = plot_search_trajectory_single_objective_hpo(
        results, mode="min", x_units="seconds", ax=axes[0],
    )

    _ = plot_worker_utilization(
        results, num_workers=num_workers, profile_type="start/end", ax=axes[1],
    )

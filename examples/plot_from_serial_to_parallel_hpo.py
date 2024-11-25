# -*- coding: utf-8 -*-
"""
From Serial to Parallel Evaluations
===================================

**Author(s)**: Romain Egele.

This example demonstrates the advantages of parallel evaluations over sequential
evaluations. We start by defining an artificial black-box ``run``-function by
using the Ackley function:

.. image:: https://www.sfu.ca/~ssurjano/ackley.png
  :width: 400
  :alt: Ackley Function in 2D

We will use the ``time.sleep`` function to simulate a budget of 2 secondes of
execution in average which helps illustrate the advantage of parallel
evaluations. The ``@profile`` decorator is useful to collect starting/ending
time of the ``run``-function execution which help us know exactly when we are
inside the black-box. When using this decorator, the ``run``-function will
return a dictionnary with 2 new keys ``"timestamp_start"`` and
``"timestamp_end"``. The ``run``-function is defined in a separate module
because of the "multiprocessing" backend that we are using in this example.

.. literalinclude:: ../../examples/black_box_util.py
   :language: python

After defining the black-box we can continue with the definition of our main script:
"""

import black_box_util as black_box
import matplotlib.pyplot as plt

from deephyper.analysis import figure_size
from deephyper.analysis.hpo import plot_search_trajectory_single_objective_hpo
from deephyper.evaluator import Evaluator
from deephyper.evaluator.callback import TqdmCallback
from deephyper.hpo import HpProblem, CBO

# %%
# Then we define the variable(s) we want to optimize. For this problem we
# optimize Ackley in a 2-dimensional search space, the true minimul is
# located at ``(0, 0)``.

nb_dim = 2
problem = HpProblem()
for i in range(nb_dim):
    problem.add_hyperparameter((-32.768, 32.768), f"x{i}")
problem

# %%
# Then we define sequential search by creation a ``"sequential"``-evaluator and we
# execute the search with a fixed time-budget of 2 minutes (i.e., 120
# secondes).

if __name__ == "__main__":
    # we give a budget of 2 minutes for each search
    timeout = 120
    sequential_evaluator = Evaluator.create(
        black_box.run_ackley,
        method="thread",  # because the ``run_function`` is not asynchronous
        method_kwargs={"callbacks": [TqdmCallback()]},
    )
    print("Running sequential search...")
    results = {}
    sequential_search = CBO(problem, sequential_evaluator, random_state=42)
    results["sequential"] = sequential_search.search(timeout=timeout)
    results["sequential"]["m:timestamp_end"] = (
        results["sequential"]["m:timestamp_end"]
        - results["sequential"]["m:timestamp_start"].iloc[0]
    )

# %%
# After, executing the sequential-search for 2 minutes we can create a parallel
# search which uses the ``"process"``-evaluator and defines 5 parallel
# workers. The search is also executed for 2 minutes.

if __name__ == "__main__":
    parallel_evaluator = Evaluator.create(
        black_box.run_ackley,
        method="thread",
        method_kwargs={"num_workers": 5, "callbacks": [TqdmCallback()]},
    )
    print("Running parallel search...")
    parallel_search = CBO(
        problem, parallel_evaluator, multi_point_strategy="qUCBd", random_state=42
    )
    results["parallel"] = parallel_search.search(timeout=timeout)
    results["parallel"]["m:timestamp_end"] = (
        results["parallel"]["m:timestamp_end"]
        - results["parallel"]["m:timestamp_start"].iloc[0]
    )

# %%
# Finally, we plot the results from the collected DataFrame. The execution
# time is used as the x-axis which help-us vizualise the advantages of the
# parallel search.

if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=figure_size(width=600))

    for strategy, df in results.items():
        plot_search_trajectory_single_objective_hpo(
            df,
            show_failures=False,
            mode="min",
            x_units="seconds",
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

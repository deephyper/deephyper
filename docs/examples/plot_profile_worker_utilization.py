# -*- coding: utf-8 -*-
"""
Profile the Worker Utilization
==============================

**Author(s)**: Romain Egele.

This example demonstrates the advantages of parallel evaluations over serial
evaluations. We start by defining an artificial black-box ``run``-function by
using the Ackley function:

.. image:: https://www.sfu.ca/~ssurjano/ackley.png
  :width: 400
  :alt: Ackley Function in 2D

We will use the ``time.sleep`` function to simulate a budget of 2 secondes of
execution in average which helps illustrate the advantage of parallel
evaluations. The ``@profile`` decorator is useful to collect starting/ending
time of the ``run``-function execution which help us know exactly when we are
inside the black-box. This decorator is necessary when profiling the worker
utilization. When using this decorator, the ``run``-function will return a
dictionnary with 2 new keys ``"timestamp_start"`` and ``"timestamp_end"``.
The ``run``-function is defined in a separate module because of
the "multiprocessing" backend that we are using in this example.

.. literalinclude:: ../../examples/black_box_util.py
   :language: python
   :emphasize-lines: 19-28
   :linenos:

After defining the black-box we can continue with the definition of our main script:
"""

import black_box_util as black_box
import matplotlib.pyplot as plt

from deephyper.analysis import figure_size
from deephyper.analysis.hpo import (
    plot_search_trajectory_single_objective_hpo,
    plot_worker_utilization,
)
from deephyper.evaluator import Evaluator
from deephyper.evaluator.callback import TqdmCallback
from deephyper.hpo import CBO, HpProblem

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
# Then we define a parallel search.

if __name__ == "__main__":
    timeout = 20
    num_workers = 4
    results = {}

    evaluator = Evaluator.create(
        black_box.run_ackley,
        method="process",
        method_kwargs={
            "num_workers": num_workers,
            "callbacks": [TqdmCallback()],
        },
    )
    search = CBO(
        problem,
        evaluator,
        random_state=42,
    )
    results = search.search(timeout=timeout)

# %%
# Finally, we plot the results from the collected DataFrame.

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
    )

    plot_search_trajectory_single_objective_hpo(
        results, mode="min", x_units="seconds", ax=axes[0]
    )

    plot_worker_utilization(
        results, num_workers=num_workers, profile_type="start/end", ax=axes[1]
    )

    plt.tight_layout()
    plt.show()

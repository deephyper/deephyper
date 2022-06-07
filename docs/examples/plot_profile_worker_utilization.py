# -*- coding: utf-8 -*-
"""
Profile the Worker Utilization
==============================

**Author(s)**: Romain Egele.

This example demonstrates the advantages of parallel evaluations over serial evaluations. We start by defining an artificial black-box ``run``-function by using the Ackley function:

.. image:: https://www.sfu.ca/~ssurjano/ackley.png
  :width: 400
  :alt: Ackley Function in 2D

We will use the ``time.sleep`` function to simulate a budget of 2 secondes of execution in average which helps illustrate the advantage of parallel evaluations. The ``@profile`` decorator is useful to collect starting/ending time of the ``run``-function execution which help us know exactly when we are inside the black-box. This decorator is necessary when profiling the worker utilization. When using this decorator, the ``run``-function will return a dictionnary with 2 new keys ``"timestamp_start"`` and ``"timestamp_end"``. The ``run``-function is defined in a separate module because of the "multiprocessing" backend that we are using in this example.

.. literalinclude:: ../../examples/black_box_util.py
   :language: python
   :emphasize-lines: 19-28 
   :linenos:

After defining the black-box we can continue with the definition of our main script:
"""
import black_box_util as black_box


# %%
# Then we define the variable(s) we want to optimize. For this problem we optimize Ackley in a 2-dimensional search space, the true minimul is located at ``(0, 0)``.
from deephyper.problem import HpProblem


nb_dim = 2
problem = HpProblem()
for i in range(nb_dim):
    problem.add_hyperparameter((-32.768, 32.768), f"x{i}")
problem


# %%
# Then we define a parallel search.
if __name__ == "__main__":
    from deephyper.evaluator import Evaluator
    from deephyper.evaluator.callback import TqdmCallback
    from deephyper.search.hps import CBO

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
    search = CBO(problem, evaluator, random_state=42)
    results = search.search(timeout=timeout)

# %%
# Finally, we plot the results from the collected DataFrame.
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    def compile_profile(df):
        """Take the results dataframe as input and return the number of jobs running at a given timestamp."""
        history = []

        for _, row in df.iterrows():
            history.append((row["timestamp_start"], 1))
            history.append((row["timestamp_end"], -1))

        history = sorted(history, key=lambda v: v[0])
        nb_workers = 0
        timestamp = [0]
        n_jobs_running = [0]
        for time, incr in history:
            nb_workers += incr
            timestamp.append(time)
            n_jobs_running.append(nb_workers)

        return timestamp, n_jobs_running

    plt.figure()

    plt.subplot(2, 1, 1)
    plt.scatter(results.timestamp_end, results.objective)
    plt.plot(results.timestamp_end, results.objective.cummax())
    plt.xlabel("Time (sec.)")
    plt.ylabel("Objective")
    plt.grid()

    plt.subplot(2, 1, 2)
    x, y = compile_profile(results)
    y = np.asarray(y) / num_workers * 100

    plt.step(
        x,
        y,
        where="pre",
    )
    plt.ylim(0, 100)
    plt.xlabel("Time (sec.)")
    plt.ylabel("Worker Utilization (%)")
    plt.tight_layout()
    plt.show()

r"""
Black-Box Optimization
======================

**Author(s)**: Romain Egele, Brett Eiffert.

In this tutorial, we introduce you to the notion of `black-box optimization (Wikipedia) <https://en.wikipedia.org/wiki/Derivative-free_optimization>`_ (a.k.a., derivative-free optimization) with DeepHyper.

Black-box optimization is a field of optimization research where an objective function :math:`f(x) = y \in \mathbb{R}` is optimized only based on input-output observations :math:`\{ (x_1,y_1), \ldots, (x_n, y_n) \}`.
 
Let's start by installing DeepHyper!
"""

# %%
#
# .. code-block:: bash
#
#     %%bash
#     pip install deephyper

# %%
# Then, we can import it and check the installed version:

import deephyper
print(deephyper.__version__)

# %%
# Optimization Problem
# --------------------
# 
# The optimization problem is based  on two components:
# 
# 1. The black-box function that we want to optimize.
# 2. The search space or domain of input variables over which we want to optimize.
# 
# Black-Box Function
# ~~~~~~~~~~~~~~~~~~
#
# DeepHyper is developed to optimize black-box functions.
# Here, we define the function :math:`f(x) = - x ^ 2` that we want to maximise (the maximum being :math:`f(x=0) = 0` on :math:`I_x = [-10;10]`). The black-box function ``f`` takes as input a ``job`` that follows a dictionary interface from which we retrieve the variables of interest.

# %%
def f(job):
    return -job.parameters["x"] ** 2

# %%
# Search Space of Input Variables
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# In this example, we have only one variable :math:`x` for the black-box functin :math:`f`. We empirically decide to optimize this variable $x$ on the interval :math:`I_x = [-10;10]`. To do so we use the :class:`deephyper.hpo.HpProblem` from DeepHyper and add a **real** hyperparameter by using a tuple of two floats.

# %%
from deephyper.hpo import HpProblem


problem = HpProblem()

# Define the variable you want to optimize
problem.add_hyperparameter((-10.0, 10.0), "x")

problem

# %%
# Evaluator Interface
# -------------------
# 
# DeepHyper uses an API called :class:`deephyper.evaluator.Evaluator` to distribute the computation of black-box functions and adapt to different backends (e.g., threads, processes, MPI, Ray). An ``Evaluator`` object wraps the black-box function ``f`` that we want to optimize. Then a ``method`` parameter is used to select the backend and ``method_kwargs`` defines some available options of this backend.
#
#
# .. hint:: The ``method="thread"`` provides parallel computation only if the black-box is releasing the global interpretor lock (GIL). Therefore, if you want parallelism in Jupyter notebooks you should use the Ray evaluator (``method="ray"``) after installing Ray with ``pip install ray``.
#
# It is possible to define callbacks to extend the behaviour of ``Evaluator`` each time a function-evaluation is launched or completed. In this example we use the :class:`deephyper.evaluator.callback.TqdmCallback` to follow the completed evaluations and the evolution of the objective with a progress-bar.

from deephyper.evaluator import Evaluator
from deephyper.evaluator.callback import TqdmCallback


# define the evaluator to distribute the computation
evaluator = Evaluator.create(
    f,
    method="thread",
    method_kwargs={
        "num_workers": 4,
        "callbacks": [TqdmCallback()]
    },
)

print(f"Evaluator has {evaluator.num_workers} available worker{'' if evaluator.num_workers == 1 else 's'}")

# %%
# Search Algorithm
# ----------------
# 
# The next step is to define the search algorithm that we want to use. Here, we choose :class:`deephyper.hpo.CBO` (Centralized Bayesian Optimization) which is a sampling based Bayesian optimization strategy. This algorithm has the advantage of being asynchronous thanks to a constant liar strategy which is crutial to keep a good utilization of the resources when the number of available workers increases.

from deephyper.hpo import CBO

# define your search
search = CBO(
    problem,
    evaluator,
    acq_func="UCB",  # Acquisition function to Upper Confidence Bound
    multi_point_strategy="qUCB",  # Fast Multi-point strategy with q-Upper Confidence Bound
    n_jobs=2,  # Number of threads to fit surrogate models in parallel
)

# %%
# Then, we can execute the search for a given number of iterations by using the ``search.search(max_evals=...)``. It is also possible to use the ``timeout`` parameter if one needs a specific time budget (e.g., restricted computational time in machine learning competitions, allocation time in HPC).

# %%
results = search.search(max_evals=100)

# %%
# Finally, let us visualize the results. The ``search(...)`` returns a DataFrame also saved locally under ``results.csv`` (in case of crash we don't want to lose the possibly expensive evaluations already performed).
# 
# The DataFrame contains as columns:
#
# 1. the optimized hyperparameters: such as :math:`x` with name ``p:x``.
# 2. the ``objective`` **maximised** which directly match the results of the :math:`f` function in our example.
# 3. the ``job_id`` of each evaluated function (increased incrementally following the order of created evaluations).
# 4. the time of creation/collection of each task ``timestamp_submit`` and ``timestamp_gather`` respectively (in secondes, since the creation of the Evaluator).

# %%
results

# %%
# To get the parameters at the observed maximum value we can use the :func:`deephyper.analysis.hpo.parameters_at_max`:

from deephyper.analysis.hpo import parameters_at_max


parameters, objective = parameters_at_max(results)
print("\nOptimum values")
print("x:", parameters["x"])
print("objective:", objective)

# %%
# We can also plot the evolution of the objective to verify that we converge correctly toward :math:`0`.

import matplotlib.pyplot as plt
from deephyper.analysis.hpo import plot_search_trajectory_single_objective_hpo


WIDTH_PLOTS = 8
HEIGHT_PLOTS = WIDTH_PLOTS / 1.618

fig, ax = plt.subplots(figsize=(WIDTH_PLOTS, HEIGHT_PLOTS))
plot_search_trajectory_single_objective_hpo(results, mode="min", ax=ax)
_ = plt.title("Search Trajectory")
_ = plt.yscale("log")

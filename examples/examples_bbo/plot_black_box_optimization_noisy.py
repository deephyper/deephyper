r"""
Noisy Black-Box Optimization
============================

**Author(s)**: Romain Egele.

In this tutorial, we show you how to manage **noisy** `black-box optimization (Wikipedia) <https://en.wikipedia.org/wiki/Derivative-free_optimization>`_ (a.k.a., derivative-free optimization) with DeepHyper.

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
# Optimization Problem
# --------------------
#
# The optimization problem consists of two components:
#
# 1. The *black-box function* that we aim to optimize.
# 2. The *search space* (or domain) of input variables over which the optimization is performed.
#
# Black-Box Function
# ~~~~~~~~~~~~~~~~~~
#
# In DeepHyper, black-box optimization is performed on user-defined functions that can be noisy or stochastic.
# Below, we define a noisy black-box function `f` that depends on a single variable :math:`x` in the domain
# :math:`I_x = [-10, 10]`.
#
# The noisy black-box function is defined as:
#
# .. math::
#     f(x) = \text{Binomial}(n=1, p(x))
#
# where the probability of success is:
#
# .. math::
#     p(x) = \frac{100 - x^2}{100}.
#
# This means that for each evaluation, `f(x)` returns a random binary value (0 or 1) with probability `p(x)` of success.
# The maximum expected value of :math:`f(x)` is obtained at :math:`x = 0`, where :math:`p(0) = 1`.
#
# The function `f` takes as input a `job`, which behaves like a dictionary.
# The variable of interest `x` is accessed via `job.parameters["x"]`.

# %%
import numpy as np


def f(job):
    p = (100 - job.parameters["x"] ** 2) / 100
    obs = np.random.binomial(n=1, p=p)
    return obs

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
        "num_workers": 1,
        "callbacks": [TqdmCallback()]
    },
)

print(f"Evaluator has {evaluator.num_workers} available worker{'' if evaluator.num_workers == 1 else 's'}")

# %%
# Search Algorithm
# ----------------
# 
# The next step is to define the search algorithm that we want to use. Here, we choose :class:`deephyper.hpo.CBO` (Centralized Bayesian Optimization) which is a sampling based Bayesian optimization strategy. 
# This algorithm has the advantage of being asynchronous which is crutial to keep a good utilization of the resources when the number of available workers increases.
# We also choose, how to optimize the acquisition function of the Bayesian optimization with ``"ga"`` (i.e., continuous Genetic Algorithm).
#
# Then, we setup a solution selection method. Here we use :class:`deephyper.hpo.ArgMaxEstSelection`, that will select the optimum based on the estimated maximum of a surrogate model.
# The ``model_grid_search=True`` activates the auto-tuning of the surrogate model every 100 observations by default.
# The ``noisy_objective=True`` sets the default configuration of the surrogate model for a noisy objective.

from deephyper.hpo import CBO, ArgMaxEstSelection


def create_search():
    search = CBO(
        problem,
        acq_optimizer="ga",
        solution_selection=ArgMaxEstSelection(
            problem,
            model_grid_search=True,
            noisy_objective=True,
        ),
    )
    return search

max_evals = 300
search = create_search()
results = search.search(evaluator, max_evals)

# %%
# Finally, let us visualize the results. The ``search(...)`` returns a DataFrame also saved locally under ``results.csv`` (in case of crash we don't want to lose the possibly expensive evaluations already performed).
# 
# The DataFrame contains the usual columns:
#
# 1. the optimized hyperparameters: such as :math:`x` with name ``p:x``.
# 2. the ``objective`` **maximised** which directly match the results of the :math:`f` function in our example.
# 3. the ``job_id`` of each evaluated function (increased incrementally following the order of created evaluations).
# 4. the time of creation/collection of each task ``timestamp_submit`` and ``timestamp_gather`` respectively (in secondes, since the creation of the Evaluator).
#
# In addition, it now also contains the new columns:
# 1. the estimated solution parameter ``sol.p:x``.
# 2. the estimated solution objective ``sol.objective``.
# 3. the estimated solution objective aleatoric uncertainty ``sol.objective_std_al``.
# 4. the estimated solution objective epistemic uncertainty ``sol.objective_std_ep``.

results

# %%
# To get the parameters at the observed maximum value we can use the :func:`deephyper.analysis.hpo.parameters_at_max`:
# We make sure to select the right column and prefix for parameters.
# Also, we prefer to select the solution amoung the ``n_last=20`` rows to avoid selecting noisy observations at the beginning.
from deephyper.analysis.hpo import parameters_at_max


parameters, objective = parameters_at_max(results, column="sol.objective", prefix="sol.p:", n_last=20)
print("\nEstimated Optimum values")
print("x:", parameters["x"])
print("objective:", objective)

# %%
# We can also plot the evolution of the estimated solution value of :math:`x` to verify that we converge correctly toward :math:`x=0`.

import matplotlib.pyplot as plt
from deephyper.analysis.hpo import plot_search_trajectory_single_objective_hpo


WIDTH_PLOTS = 8
HEIGHT_PLOTS = WIDTH_PLOTS / 1.618

fig, ax = plt.subplots(figsize=(WIDTH_PLOTS, HEIGHT_PLOTS))
plot_search_trajectory_single_objective_hpo(results, column="sol.p:x", mode="max", ax=ax)
_ = ax.set_ylabel(r"Estimated solution $x$")
_ = ax.set_ylim(-10, 10)
_ = plt.title("Search Trajectory")
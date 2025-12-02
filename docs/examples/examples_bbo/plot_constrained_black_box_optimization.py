r"""
Constrained Black-Box Optimization with Rejection Sampling
==========================================================

**Author(s)**: Romain Egele.

In this tutorial, we illustrate how to solve constrained `black-box optimization (Wikipedia) <https://en.wikipedia.org/wiki/Derivative-free_optimization>`_ (also known as derivative-free optimization) using DeepHyper.

Black-box optimization refers to a class of methods where an objective function :math:`f(x) = y \in \mathbb{R}` can only be queried through input–output evaluations :math:`\{ (x_1, y_1), \ldots, (x_n, y_n) \}`. No closed-form expression, derivatives, or structural information about :math:`f` are required.

In *constrained* optimization, we further introduce one or more rules that restrict the set of admissible solutions. These constraints carve out the feasible region of the search space and can substantially influence both the behavior and performance of the optimizer.

In the following example, we define a simple two-dimensional problem, impose a linear constraint, and solve it using DeepHyper’s Centralized Bayesian Optimization (CBO) engine.

There exists multiple ways of handling constraints in DeepHyper:

#. Rejection sampling (shown in this tutorial) where we sample from the unconstrained search space 
   then reject unfeasible solutions. While this approach is simple it can become computationnaly 
   intractable.
#. Learning to avoid failures (see :ref:`Learn to Avoid Failures with Bayesian Optimization <sphx_glr_examples_examples_bbo_plot_notify_failures_hpo.py>`).
#. Define the constraint as an other objective in a multi-objective optimization setup (tutorial coming soon).
#. Custom chained sampler (tutorial coming soon).
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from deephyper.analysis.hpo import plot_search_trajectory_single_objective_hpo, parameters_at_max
from deephyper.hpo import HpProblem, CBO

# %%
# Optimization Problem
# --------------------
# We define a 2D search space over variables ``x`` and ``y``.

pb = HpProblem()
pb.add((-10.0, 10.0), "x")
pb.add((-10.0, 10.0), "y")


def f(job):
    """Objective function: maximize x^2 + y^2."""
    return job.parameters["x"] ** 2 + job.parameters["y"] ** 2


def constraint_fn_test(s: pd.DataFrame):
    """Feasibility condition: |x| + |y| <= 10."""
    return np.abs(s["x"]) + np.abs(s["y"]) <= 10


pb.set_constraint_fn(constraint_fn_test)

# %%
# We now set up a constrained Bayesian optimization search using a genetic algorithm
# to optimize the acquisition function periodically.

search = CBO(
    pb,
    acq_optimizer="ga",
    acq_optimizer_kwargs={"acq_optimizer_freq": 2},
    acq_func_kwargs={
        # Exploration/Exploitation mechanism
        "kappa": 200.0,
        "scheduler": {"type": "periodic-exp-decay", "period": 20, "kappa_final": 1.96},
    },
    verbose=1,
)
results = search.search(f, max_evals=300)

# %%
results

# %%
# Extracting the Best Parameters
# ------------------------------
# To recover the parameters corresponding to the best observed objective value,
# we can use :func:`deephyper.analysis.hpo.parameters_at_max`.

parameters, objective = parameters_at_max(results)
print("\nOptimum values")
print(f"x: {parameters['x']:.3f}, y: {parameters['y']:.2f}")
print("objective:", objective)

# %%
# Visualizing the Search Trajectory
# ---------------------------------
# We plot the evolution of the best objective value to verify that optimization
# progresses correctly toward the maximum :math:`100`.
#
# We clearly see the periodic exploration/exploitation effect of the scheduler.

WIDTH_PLOTS = 8
HEIGHT_PLOTS = WIDTH_PLOTS / 1.618

fig, ax = plt.subplots(figsize=(WIDTH_PLOTS, HEIGHT_PLOTS))
plot_search_trajectory_single_objective_hpo(results, mode="max", ax=ax)
_ = plt.title("Search Trajectory")

# %%
# Visualizing the Feasible Region and Evaluations
# -----------------------------------------------
# We now plot all evaluated points in the (x, y) plane, color-coded by
# objective value, along with the constraint boundary ``x + y = 10``.


# sphinx_gallery_thumbnail_number = 2
fig, ax = plt.subplots(figsize=(WIDTH_PLOTS, HEIGHT_PLOTS))
item = ax.scatter(results["p:x"], results["p:y"], c=results["objective"], label="Evaluations")
ax.plot([0, 10], [10, 0], "r:")
ax.plot([0, 10], [-10, 0], "r:")
ax.plot([-10, 0], [0, -10], "r:")
ax.plot([0, -10], [10, 0], "r:", label="Constraint")
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.legend()
ax.grid()
ax.grid(which="minor", linestyle=":")
cb = plt.colorbar(item)
cb.set_label(r"Objective")
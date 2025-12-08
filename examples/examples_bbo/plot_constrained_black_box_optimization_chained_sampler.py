r"""
Constrained Black-Box Optimization with Custom Chained Sampler
==============================================================

**Author(s)**: Romain Egele.

This tutorial demonstrates how to solve *constrained*
`black-box optimization <https://en.wikipedia.org/wiki/Derivative-free_optimization>`_
using **DeepHyper**, focusing on how to encode structural constraints directly
in the **sampling strategy** of the search algorithm.

Black-box optimization aims to optimize an unknown function
:math:`f(x) = y \in \mathbb{R}` using only input–output evaluations
:math:`\{(x_1, y_1), \ldots, (x_n, y_n)\}`.
No analytical gradients or structural properties of :math:`f` are required.

In *constrained* settings, the search is further restricted to parameters that
satisfy one or more feasibility rules. Constraints can significantly reshape
the search space and modify the behavior of the optimizer.

Problem Setting
---------------
In this example, we consider a *discrete*, *ordered* search space of
dimension :math:`N`.
Each variable :math:`x_i` must satisfy the **monotonicity constraint**

.. math::

    x_0 < x_1 < \cdots < x_{N-1}.

Each :math:`x_i` is bounded between :math:`i` and :math:`m - N + i`.
This could tipically represent the layer indexes to drop in Depth pruning of Large language models.
The objective is to **maximize** the sum

.. math::

   f(x) = \sum_{i=0}^{N-1} x_i.

Since the optimal strategy is to push every variable as high as possible while
respecting monotonicity, the theoretical optimum is:

.. math::

   \sum_{i=0}^{N-1} (m - i).

DeepHyper offers several ways to incorporate constraints:

#. **Custom chained sampler** *(this tutorial)*: constraints are enforced
   directly when generating new candidate points.

#. **Rejection sampling**:
   see :ref:`Constrained Black-Box Optimization with Rejection Sampling <sphx_glr_examples_examples_bbo_plot_constrained_black_box_optimization.py>`.

#. **Learning to avoid failures** (CBO auto-handles failed evaluations):
   see this tutorial and also
   :ref:`Learn to Avoid Failures with Bayesian Optimization <sphx_glr_examples_examples_bbo_plot_notify_failures_hpo.py>`.

#. **Multi-objective approach** where the constraint becomes an additional
   objective (tutorial forthcoming).
"""

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import numpy as np
import pandas as pd

from deephyper.analysis.hpo import (
    plot_search_trajectory_single_objective_hpo,
    parameters_at_max,
    filter_failed_objectives,
)
from deephyper.hpo import HpProblem, CBO

# %%
# Custom Chained Sampler
# ----------------------
#
# Because every :math:`x_i` must be strictly larger than :math:`x_{i-1}`, the
# usual independent sampling over each variable would frequently violate the
# constraint.
#
# Instead, we implement a *chained* sampler:
# each :math:`x_k` is sampled conditionally so that enough "room" remains for
# future variables. This ensures that:
#
# - all generated samples satisfy :math:`x_i < x_{i+1}` by construction;
# - the sampler focuses on the feasible region, avoiding wasted evaluations.

n = 10
m = 32

print("optimum:", sum([m - i - 1 for i in range(n)]))

pb = HpProblem()
for i in range(n):
    pb.add((i, m - n + i - 1), f"x{i}")


def sampling_fn(size: int) -> list[dict]:
    def sample_chain():
        # Chain the sampling
        vals = []
        lo = 0
        for k in range(n):
            low = max(k, lo + 1 if k > 0 else 0)
            high = m - n + k
            v = np.random.randint(low, high - (n - 1 - k))  # keep room for future vars
            vals.append(v)
            lo = v
        return {k: v for k, v in zip(pb.hyperparameter_names, vals)}

    return [sample_chain() for _ in range(size)]


pb.set_sampling_fn(sampling_fn)

# %%
# Constraint Function
# -------------------
#
# Although the sampler already generates feasible points, we explicitly define a
# ``constraint_fn``. This allows DeepHyper to properly handle *failed* trials
# (e.g., from manually constructed parameter sets or mutation-based acquisition
# optimizers).
# Not only that, this will help report non-feasible points using ``"F_constraint"``
# in the objective function so that CBO learns to avoid them.


def constraint_fn(df: pd.DataFrame) -> pd.Series:
    accept = pd.Series(np.ones((len(df)), dtype=bool))
    for i in range(n - 1):
        accept = accept & (df[f"x{i}"] < df[f"x{i + 1}"])
    return accept


pb.set_constraint_fn(constraint_fn)


def f(job):
    """Objective function: maximize sum(x_i)."""
    df = pd.DataFrame([job.parameters])
    accept = constraint_fn(df)
    if all(accept):
        return sum(job.parameters.values())
    else:
        return "F_constraint"


# %%
# Bayesian Optimization with Mixed-GA Acquisition Optimization
# ------------------------------------------------------------
#
# We run a **Centralized Bayesian Optimization (CBO)** search using:
#
# - Ensemble of Trees surrogate model (``"ET"``).
# - A **mixed genetic algorithm** (``"mixedga"``) to optimize the acquisition
#   function.
# - A **periodically decaying scheduler** on the exploration parameter ``kappa``.
#
# This setup is well suited for discrete, irregularly constrained spaces.
#
# The search runs for ``max_evals=300`` iterations.

search = CBO(
    pb,
    surrogate_model="ET",
    surrogate_model_kwargs={"max_features": "sqrt"},
    acq_optimizer="mixedga",
    acq_optimizer_kwargs={
        "n_points": 1_000,
        "acq_optimizer_freq": 2,
        "filter_failures": "mean",
    },
    acq_func_kwargs={
        # Exploration/Exploitation mechanism
        "kappa": 10.0,
        "scheduler": {
            "type": "periodic-exp-decay",
            "period": 20,
            "kappa_final": 0.1,
        },
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
for i in range(n):
    print(f"x{i}: {parameters[f'x{i}']:.3f}")
print("objective:", objective)

# %%
# Visualization
# ---------------
# We conclude with:
#
# - a **search trajectomakery plot** showing the best objective value over time,
#   where the periodic exploration schedule is clearly visible;
#
# - a **feasible-space evaluation plot** showing all sampled curves
#   :math:`i \mapsto x_i` (each curve is one evaluation), colored by objective
#   value.
#
# These visualizations confirm that the optimizer progressively learns the
# structure of the monotonic constraint and approaches the theoretical optimum.

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


results, _ = filter_failed_objectives(results)

p_columns = [col for col in results.columns if col.startswith("p:")]

# Create a normalizer over the objective range
obj_vals = results["objective"]
norm = colors.Normalize(vmin=obj_vals.min(), vmax=obj_vals.max())

# Choose a colormap (viridis is a good default)
cmap = plt.get_cmap("viridis")

fig, ax = plt.subplots(figsize=(WIDTH_PLOTS, HEIGHT_PLOTS))

for i, row in results.iterrows():
    x_values = row[p_columns].values
    y_values = np.arange(n)
    obj_value = row["objective"]

    color = cmap(norm(obj_value))  # map objective → color
    ax.plot(x_values, y_values, color=color, alpha=0.9)

# Optionally add a colorbar
sm = cm.ScalarMappable(norm=norm, cmap=cmap)
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label("Objective value")
ax.grid()
ax.set_ylim(0, n - 1)
ax.set_xlim(0, m)
ax.set_ylabel(r"$i$")
ax.set_xlabel(r"$x_i$")
ax.set_yticks(list(range(n)), [str(i) for i in range(n)])
ax.set_xticks(list(range(0, m, 2)), [str(i) for i in range(0, m, 2)])

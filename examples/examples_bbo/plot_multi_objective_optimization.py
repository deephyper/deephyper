r"""
Mutli-Objective Black-Box Optimization
======================================

In this tutorial, we will explore how to run black-box multi-objective optimization (MOO). In this setting, the goal is to resolve the following problem:

.. math::
   \text{max}_x (f_0(x), f_1(x), ..., f_n(x))

where :math:`x` is the set of optimized variables and :math:`f_i` are the different objectives. In DeepHyper, we use scalarization to transform such multi-objective problem into a single-objective problem:

.. math::
   \text{max}_x s_w((f_0(x), f_1(x), ..., f_n(x)))

where :math:`w` is a set of weights which manages the trade-off between objectives and :math:`s_w : \mathbb{R}^n \rightarrow \mathbb{R}`. The weight vector :math:`w` is randomized and re-sampled for each new batch of suggestion from the optimizer.

We will look at the DTLZ benchmark suite, a classic in multi-objective optimization (MOO) litterature. This benchmark exibit some characteristic cases of MOO. By default, this tutorial is loading the DTLZ-II benchmark which exibit a Pareto-Front with a concave shape.
"""
# %%
# Installation and imports
# ------------------------
#
# Installing dependencies with the :ref:`pip installation <install-pip>` is recommended. It requires **Python >= 3.10**.
#
# .. code-block:: bash
#
#     %%bash
#     pip install deephyper
#     pip install -e "git+https://github.com/deephyper/benchmark.git@main#egg=deephyper-benchmark"

# .. dropdown:: Import statements
import matplotlib.pyplot as plt

from deephyper.hpo import CBO
from deephyper_benchmark.benchmarks.dtlz import DTLZBenchmark

WIDTH_PLOTS = 8
HEIGHT_PLOTS = WIDTH_PLOTS / 1.618

n_objectives = 2
bench = DTLZBenchmark(nobj=n_objectives)

# %%
# We can display the variable search space of the benchmark we just loaded:

bench.problem

# %%
# To define a black-box for multi-objective optimization it is very similar to single-objective optimization at the difference that the ``objective`` can now be a list of values. A first possibility is:
# 
# .. code-block:: python
#
#     def run(job):
#         ...
#         return objective_0, objective_1, ..., objective_n
# 
# which just returns the objectives to optimize as a tuple. If additionnal metadata are interesting to gather for each evaluation it is also possible to return them by following this format:
# 
# .. code-block:: python
#
#     def run(job):
#         ...
#         return {
#             "objective": [objective_0, objective_1, ..., objective_n],
#             "metadata": {
#                 "flops": ...,
#                 "memory_footprint": ...,
#                 "duration": ...,
#              }
#          }
#
# each of the metadata needs to be JSON serializable and will be returned in the final results with a column name formatted as ``m:metadata_key`` such as ``m:duration``.

# %%
# For the search algorithm, we use the centralized Bayesian Optimization search (CBO).
# Search algorithm
#
# The arguments specific to multi-objective optimization are:
# 
# - ``moo_scalarization_strategy`` is used to specify the scalarization strategy. 
#   Chebyshev  scalarizationis capable of generating a diverse set of solutions for non-convex problems.
# - ``moo_scalarization_weight`` argument is used to specify the weight of objectives in the scalarization.
#   ``"random"`` is used to generate a random weight vector at each iteration.

search = CBO(
    bench.problem,
    bench.run_function,
    acq_optimizer="sampling",
    moo_scalarization_strategy="AugChebyshev",
    moo_scalarization_weight="random",
    verbose=1,
)

# %%
# Launch the search for a given number of evaluations
# other stopping criteria can be used (e.g. timeout, early-stopping/convergence)
results = search.search(max_evals=500)

# %%
# A Pandas table of results is returned by the search and also saved at ``./results.csv``. An other location can be specified by using ``CBO(..., log_dir=...)``.

results

# %%
# In this table we retrieve:
# 
# - columns starting by ``p:`` which are the optimized variables.
# - the ``objective_{i}`` are the objectives returned by the black-box function.
# - the ``job_id`` is the identifier of the executed evaluations.
# - columns starting by ``m:`` are metadata returned by the black-box function.
# - ``pareto_efficient`` is a column only returned for MOO which specify if the evaluation is part of the set of optimal solutions.
#

# %%
# Let us use this table to visualize evaluated objectives.
# The estimated optimal solutions will be colored in red.

# .. dropdown:: Plot evaluated objectives
fig, ax = plt.subplots(figsize=(WIDTH_PLOTS, HEIGHT_PLOTS), tight_layout=True)
_ = ax.plot(
    -results[~results["pareto_efficient"]]["objective_0"],
    -results[~results["pareto_efficient"]]["objective_1"],
    "o",
    color="blue",
    alpha=0.7,
    label="Non Pareto-Efficient",
)
_ = ax.plot(
    -results[results["pareto_efficient"]]["objective_0"],
    -results[results["pareto_efficient"]]["objective_1"],
    "o",
    color="red",
    alpha=0.7,
    label="Pareto-Efficient",
)
_ = ax.grid()
_ = ax.legend()
_ = ax.set_xlabel("Objective 0")
_ = ax.set_ylabel("Objective 1")

# %%
# Let us look the evolution of the hypervolume indicator.
# This metric should increase over time.

# .. dropdown:: Plot hypervolume
scorer = bench.scorer
hvi = scorer.hypervolume(results[["objective_0", "objective_1"]].values)
x = list(range(1, len(hvi)+1))
fig, ax = plt.subplots(figsize=(WIDTH_PLOTS, HEIGHT_PLOTS), tight_layout=True)
_ = ax.plot(x, hvi)
_ = ax.grid()
_ = ax.set_xlabel("Evaluations")
_ = ax.set_ylabel("Hypervolume Indicator")
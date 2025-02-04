r"""
Standard Experimental Design with Grid Search
=============================================

**Author(s)**: Romain Egele.

This example demonstrates how to generate points from standard experimental
designs (e.g., random, grid, lhs).

"""
# .. dropdown:: Import statements
import os
import shutil

import matplotlib.pyplot as plt

from deephyper.analysis._matplotlib import update_matplotlib_rc
from deephyper.hpo import HpProblem
from deephyper.hpo import ExperimentalDesignSearch


update_matplotlib_rc()

# %%
# First we define the hyperparameter search space.

problem = HpProblem()
problem.add_hyperparameter((0.0001, 100.0, "log-uniform"), "x")
problem.add_hyperparameter((0.0, 100.0), "y")
problem.add_hyperparameter([1, 2, 3], "z")
problem

# %%
# Then we define the black-box function to optimize.

def run(job):
    config = job.parameters
    objective = config["x"] + config["y"]
    return objective

# %%

# .. dropdown:: Clean up legacy results
log_dir = "eds_logs"
if os.path.exists(log_dir):
    shutil.rmtree(log_dir)

# %%
# Then we define the search. In this example, we use the
# `ExperimentalDesignSearch` class to generate points from a grid design. The
# `Evaluator` can also be used with this class to parallelize evalutions.
# Note that `n_points` and `max_evals` take the same value here.

max_evals = 200
search = ExperimentalDesignSearch(
    problem, 
    run, 
    n_points=max_evals, 
    design="grid", 
    log_dir=log_dir,
)
results = search.search(max_evals)

# %%
# Finally, we plot the results from the collected DataFrame.

# .. dropdown:: Make plot
fig, ax = plt.subplots()
ax.scatter(results["p:x"], results["p:y"], c=results["p:z"], alpha=0.3)
ax.set_xscale("log")
_ = plt.xlabel("x")
_ = plt.ylabel("y")

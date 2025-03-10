r"""
Notify Failures in Hyperparameter optimization
==============================================

**Author(s)**: Romain Egele.

In this example, you will learn how to handle failures in black-box optimization.
In many application of black-box optimization such as software auto-tuning (where we
minimize the run-time of a software application) some configurations can
create run-time errors and therefore no scalar objective is returned. A
default choice could be to return in this case the worst case objective if
known. Other possibilites are to ignore these configurations or to replace 
them with the running average or minimum objective.
"""

# %%

# .. dropdown:: Import statements
import matplotlib.pyplot as plt
import numpy as np

from deephyper.hpo import HpProblem
from deephyper.hpo import CBO
from deephyper.evaluator import Evaluator
from deephyper.evaluator.callback import TqdmCallback

WIDTH_PLOTS = 8
HEIGHT_PLOTS = WIDTH_PLOTS / 1.618

# %%
# To illustrate such a use-case we define a ``run``-function that will fail when its 
# input parameter ``p_failure`` is greater than 0.5.
# To notify deephyper about a failure, we return a "string" value with ``"F"`` as prefix such as:
def run(job) -> float:
    if job.parameters["p_failure"] > 0.5:
        try:
            raise ValueError("Some example exception")
        except ValueError:
            # Notify the search about the failure
            return "F_value_error"
    else:
        # Returns a regular objective value that is maximized
        return sum(job.parameters[k] for k in job.parameters if "x" in k)


# %%
# Then, we define the corresponding hyperparameter problem where ``x{i}`` are the
# value to maximize and ``p_parameter`` is a value that impact the appearance of failures.
problem = HpProblem()
problem.add_hyperparameter((0.0, 1.0), "p_failure")
for i in range(10):
    problem.add_hyperparameter((0.0, 1.0), f"x{i}")
problem

# %%
# We use the centralized Bayesian optimization (CBO) for the search.
# The :class:`deephyper.hpo.CBO` has a parameter ``filter_failures``.
# We will compare:
# 
# - ``filter_failures="ignore"``: filters-out failed configurations.
# - ``filter_failures="mean"``: replaces failures with the running average of non-failed objectives.
# - ``filter_failures="min"``: replaces failures with the running minimum of non-failed objectives.

results = {}
max_evals = 100

# %%
for failure_strategy in ["ignore", "mean", "min"]:

    evaluator = Evaluator.create(
        run,
        method="thread",
        method_kwargs={
            "callbacks": [TqdmCallback(f"Failure Strategy: {failure_strategy}")]}
    )

    search = CBO(
        problem,
        evaluator,
        acq_func="UCBd",
        acq_optimizer="ga",
        acq_optimizer_freq=1,
        filter_duplicated=False,
        filter_failures=failure_strategy,
        log_dir=f"search_{failure_strategy}",
        random_state=42,
    )

    results[failure_strategy] = search.search(max_evals)

# %%
# Finally we plot the collected results.
# We can see that the ``"mean"`` and ``min`` strategy have much less failures than ``"ignore"``.
# In addition, we observe that they return significantly better objectives.

# .. dropdown:: Plot results with failures
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(WIDTH_PLOTS, HEIGHT_PLOTS), tight_layout=True, sharex=True, sharey=True)
for i, (failure_strategy, df) in enumerate(results.items()):
    
    if df.objective.dtype != np.float64:
        x = np.arange(len(df))
        mask_failed = np.where(df.objective.str.startswith("F"))[0]
        mask_success = np.where(~df.objective.str.startswith("F"))[0]
        x_success, x_failed = x[mask_success], x[mask_failed]
        y_success = df["objective"][mask_success].astype(float)

    axes[i].scatter(x_success, y_success, label=failure_strategy)
    axes[i].scatter(x_failed, np.zeros(x_failed.shape), marker="v", color="red")

    axes[i].set_xlabel("Evalutions")
    axes[i].set_ylabel("Objective")
    axes[i].legend()
    axes[i].grid()


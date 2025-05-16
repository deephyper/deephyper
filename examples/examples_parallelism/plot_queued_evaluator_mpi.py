r"""
Queued Evaluator with MPI
=========================

**Author(s)**: Romain Egele, Brett Eiffert.

In this example, you will learn how to use a queued ProcessPoolEvaluator with an mpi run function.
"""

# %%

# .. dropdown:: Import statements
import subprocess

from deephyper.evaluator import (
    LokyEvaluator,
    parse_subprocess_result,
    profile,
    queued,
)
from deephyper.evaluator.callback import TqdmCallback
from deephyper.hpo import CBO, HpProblem


# %%
# Run function using MPI
# ----------------
#
# Used by the main function to fork and manage mpi processes within the evaluator.
@profile
def run_mpi_exe(job, dequed=None):
    x = job.parameters["x"]

    # The format of the print "DH-OUTPUT:(.+)\n" is strict if you use parse_suprocess_result
    command = f"mpirun -np {len(dequed)} echo DH-OUTPUT:{x}\n"
    completed_process = subprocess.run(command.split(), capture_output=True)
    objective = parse_subprocess_result(completed_process)

    # In the results.csv a new `m:dequed` will show the passed dequed values.
    return {"objective": objective, "metadata": {}}


# %%
# Setup
# -----
#
# The problem is defined with basic hyperparameters for a straightforward example.

problem = HpProblem()
problem.add_hyperparameter((0.0, 10.0), "x")

# %%
# Variables used for selecting the number of workers to execute the pool of mpi workers.
# These are defined for show and can be run in a multi-node setup or on a single node or local machine.
# Number of processes spawned = num_nodes / num_nodes_per_task

# Local machine or single node
num_nodes = 1
num_nodes_per_task = 1
# Multi-node setup
#num_nodes = 10
#num_nodes_per_task = 2

# %%
# Parallel Processing
# -------------------
#
# We define a main function which sets up an mpi enabled evaluator object to be used to evaluate the model in parallel. Tasks are spawned in the run_mpi_exe function that was defined earlier and queued in a ``LokyEvaluator``.
#
# Using the evaluator (``LokyEvaluator``), the search is performed for a user defined number of iterations (50).
# ``LokyEvaluator`` was chosen over other deephyper evaluators ``ProcessPoolEvaluator`` and ``ThreadPoolEvaluator`` due to the preference of running MPI processes and the necessity of argument based process spawning required by notebook-style runtimes.
#  To read more about the evaluator backend options and how to choose the best on for a specific use case, go to (coming soon). 

def main():
    evaluator = queued(LokyEvaluator)(
        run_function=run_mpi_exe,
        num_workers=num_nodes // num_nodes_per_task,
        callbacks=[TqdmCallback()],
        queue=[node_id for node_id in range(num_nodes)],
        queue_pop_per_task=num_nodes_per_task,
    )

    print(f"Evaluator uses {evaluator.num_workers} workers")

    search = CBO(problem, evaluator, log_dir="log_queued_evaluator")
    
    search.search(max_evals=50)

if __name__ == "__main__":
    main()

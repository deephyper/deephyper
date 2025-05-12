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
    ProcessPoolEvaluator,
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
# Used to start and manage mpi processes.
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
# Variables used for selecting the number of workers to execute the pool of mpi workers. These are defined for show and can be run in a multi-node setup or on a single node or local machine.
#
# Number of processes spawned = num_nodes / num_nodes_per_task

num_nodes = 10
num_nodes_per_task = 2

# %%
# Parallel Processing
# -------------------
#
# We define a main function which sets up an mpi enabled evaluator object to be used to evaluate the model in parallel. Tasks are spawned in the run_mpi_exe function that was defined earlier and queued in a ``ProcessPoolEvaluator``.
#
# Using evaluator (``ProcessPoolEvaluator``), the search is performed for a user defined number of iterations (50).

def main():
    evaluator = queued(ProcessPoolEvaluator)(
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

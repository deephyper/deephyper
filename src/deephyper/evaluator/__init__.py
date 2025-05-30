"""Evaluator subpackage.

This provides a common interface to execute isolated tasks with different
parallel backends and system properties.

This interface is used by search algorithm to perform black-box optimization
(the black-box being represented by the ``run``-function). An ``Evaluator``,
when instanciated, is bound to a ``run``-function which takes as first
argument a dictionnary and optionally has other keyword-arguments. The
``run``-function has to return a Python serializable value (under ``pickle``
protocol). In it's most basic form the return value is a ``float``.

An example ``run``-function is:

.. code-block:: python

    def run(job: RunningJob) -> Union[float, str, Dict]:

        config = job.parameters
        y = config["x"]**2

        return y

The return value of the ``run``-function respect the following standards
(but the feature is not necessarily supported by all search algorithms, such
as multi-objective optimization):

.. code-block:: python

    # float for single objective optimization
    return 42.0
    # str with "F" prefix for failed evaluation
    return "F_out_of_memory"
    # dict
    return {"objective": 42.0}
    # dict with additional information
    return {"objective": 42.0, "metadata": {"num_epochs_trained": 25, "num_parameters": 420000}}
    # dict with reserved keywords (when @profile decorator is used)
    return {"objective": 42.0, "metadata": {"timestamp_start": ..., "timestamp_end": ...}"
    # tuple of float for multi-objective optimization, appears as "objective_0" and "objective_1"
    return 42.0, 0.42

"""

from deephyper.evaluator._decorator import profile
from deephyper.evaluator._encoder import parse_subprocess_result, to_json
from deephyper.evaluator._evaluator import EVALUATORS, Evaluator, MaximumJobsSpawnReached
from deephyper.evaluator._job import HPOJob, Job, JobStatus, RunningJob
from deephyper.evaluator._loky import LokyEvaluator
from deephyper.evaluator._process_pool import ProcessPoolEvaluator
from deephyper.evaluator._queued import queued
from deephyper.evaluator._serial import SerialEvaluator
from deephyper.evaluator._thread_pool import ThreadPoolEvaluator

__all__ = [
    "Evaluator",
    "EVALUATORS",
    "HPOJob",
    "Job",
    "JobStatus",
    "parse_subprocess_result",
    "ProcessPoolEvaluator",
    "LokyEvaluator",
    "MaximumJobsSpawnReached",
    "profile",
    "queued",
    "RunningJob",
    "SerialEvaluator",
    "ThreadPoolEvaluator",
    "to_json",
]

try:
    from deephyper.evaluator._ray import RayEvaluator  # noqa: F401

    __all__.append("RayEvaluator")
except ImportError:
    pass

try:
    from deephyper.evaluator._mpi_comm import MPICommEvaluator  # noqa: F401

    __all__.append("MPICommEvaluator")
except ImportError:
    pass

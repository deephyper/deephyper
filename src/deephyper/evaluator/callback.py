"""The callback module contains sub-classes of the ``Callback`` class.

The ``Callback`` class is used to trigger custom actions on the start and
completion of jobs by the ``Evaluator``. Callbacks can be used with any
``Evaluator`` implementation.
"""

import abc
import csv
import logging
import os
from typing import List

import numpy as np

from deephyper.evaluator import HPOJob, Job
from deephyper.evaluator.utils import test_ipython_interpretor
from deephyper.skopt.moo import hypervolume

if test_ipython_interpretor():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

__all__ = ["Callback", "LoggerCallback", "TqdmCallback", "SearchEarlyStopping"]

logger = logging.getLogger(__name__)


class Callback(abc.ABC):
    """Callback interface."""

    def on_launch(self, job: Job):
        """Called each time a ``Job`` is created by the ``Evaluator``.

        Args:
            job (Job): The created job.
        """

    def on_done(self, job: Job):
        """Called each time a local ``Job`` has been gathered by the Evaluator.

        Args:
            job (Job): The completed job.
        """

    def on_done_other(self, job: Job):
        """Called after local ``Job`` have been gathered for each remote ``Job`` that is done.

        Args:
            job (Job): The completed Job.
        """

    def on_gather(self, local_jobs: List[Job], other_jobs: List[Job]):
        """Called after gathering jobs.

        Args:
            local_jobs (List[Job]):
                gathered jobs from local evaluator instance.

            other_jobs (List[Job]):
                gathered jobs from other evaluators using the same storage.
        """

    def on_close(self):
        """Called when the evaluator is being closed."""


class ObjectiveRecorder:
    """Records the objective values of the jobs.

    :meta: private
    """

    def __init__(self):
        self._objectives = []
        self.is_multi_objective = False
        self._last_return = -float("inf")

    def __call__(self, job):
        """Called when a local job has been gathered."""
        # Only add the objective if it is not a string (i.e., failure...)
        if not isinstance(job.objective, str):
            self._objectives.append(job.objective)

            # Then check if the objective is multi-objective
            if np.ndim(job.objective) > 0:
                self.is_multi_objective = True

        # If no objectives are received but only failures then return -inf
        if len(self._objectives) == 0:
            return self._last_return

        # If single objective then returns the maximum
        if not self.is_multi_objective:
            self._last_return = max(self._objectives[-1], self._last_return)
            return self._last_return
        else:
            objectives = -np.asarray(self._objectives)
            ref = np.max(objectives, axis=0)  # reference point
            return hypervolume(objectives, ref)


class LoggerCallback(Callback):
    """Print information when jobs are completed by the ``Evaluator``.

    An example usage can be:

    >>> evaluator.create(method="ray", method_kwargs={..., "callbacks": [LoggerCallback()]})
    """

    def __init__(self):
        self._best_objective = None
        self._n_done = 0
        self._objective_func = ObjectiveRecorder()

    def on_done_other(self, job):
        """Called after gathering local jobs on available remote jobs that are done."""
        self.on_done(job)

    def on_done(self, job):
        """Called when a local job has been gathered."""
        self._n_done += 1
        # Test if multi objectives are received
        if np.ndim(job.objective) > 0:
            if np.isreal(job.objective).all():
                self._best_objective = self._objective_func(job)

                tmp = tuple(round(o, 5) if not isinstance(o, str) else o for o in job.objective)
                print(
                    f"[{self._n_done:05d}] -- HVI Objective: {self._best_objective:.5f} -- "
                    f"Last Objective: {tmp}"
                )

            elif np.any(type(res) is str and "F" == res[0] for res in job.objective):
                print(f"[{self._n_done:05d}] -- Last Failure: {job.objective}")

        elif np.isreal(job.objective):
            self._best_objective = self._objective_func(job)

            print(
                f"[{self._n_done:05d}] -- Maximum Objective: {self._best_objective:.5f} -- "
                f"Last Objective: {job.objective:.5f}"
            )
        elif type(job.objective) is str and "F" == job.objective[0]:
            print(f"[{self._n_done:05d}] -- Last Failure: {job.objective}")


class TqdmCallback(Callback):
    """Print information when jobs are completed by the ``Evaluator``.

    Args:
        description (str, optional): an optional description to add to the progressbar.

    An example usage can be:

    >>> evaluator.create(method="ray", method_kwargs={..., "callbacks": [TqdmCallback()]})
    """

    def __init__(self, description: str = None):
        self._best_objective = None
        self._n_done = 0
        self._n_failures = 0
        self._max_evals = None
        self._tqdm = None
        self._objective_func = ObjectiveRecorder()
        self._description = description

    def set_max_evals(self, max_evals):
        """Setter for the maximum number of evaluations.

        It is used to initialize the tqdm progressbar.
        """
        self._max_evals = max_evals
        self._tqdm = None

    def on_done_other(self, job):
        """Called after gathering local jobs on available remote jobs that are done."""
        self.on_done(job)

    def on_done(self, job):
        """Called when a local job has been gathered."""
        if self._tqdm is None:
            if self._max_evals:
                self._tqdm = tqdm(total=self._max_evals)
            else:
                self._tqdm = tqdm()

            if self._description:
                self._tqdm.set_description(self._description)

        self._n_done += 1
        self._tqdm.update(1)

        if isinstance(job, HPOJob):
            # Test if multi objectives are received
            if np.ndim(job.objective) > 0:
                if not (any(not (np.isreal(objective_i)) for objective_i in job.objective)):
                    self._best_objective = self._objective_func(job)
                else:
                    self._n_failures += 1
                self._tqdm.set_postfix({"failures": self._n_failures, "hvi": self._best_objective})
            else:
                if np.isreal(job.objective):
                    self._best_objective = self._objective_func(job)
                else:
                    self._n_failures += 1
                self._tqdm.set_postfix(objective=self._best_objective, failures=self._n_failures)

        if self._max_evals == self._n_done:
            self._tqdm.close()


class SearchEarlyStopping(Callback):
    """Stop the search gracefully when it does not improve for a given number of evaluations.

    Args:
        patience (int, optional):
            The number of not improving evaluations to wait for before
            stopping the search. Defaults to ``10``.
        objective_func (callable, optional):
            A function that takes a ``Job`` has input and returns the maximized scalar value
            monitored by this callback. Defaults to computes the maximum for single-objective
            optimization and the hypervolume for multi-objective optimization.
        threshold (float, optional):
            The threshold to reach before activating the patience to stop the
            search. Defaults to ``None``, patience is reinitialized after
            each improving observation.
        verbose (bool, optional): Activation or deactivate the verbose mode. Defaults to ``True``.
    """

    def __init__(
        self,
        patience: int = 10,
        objective_func=None,
        threshold: float = None,
        verbose: bool = 1,
    ):
        self._best_objective = None
        self._n_lower = 0
        self._patience = patience
        self._objective_func = ObjectiveRecorder() if objective_func is None else objective_func
        self._threshold = threshold
        self._verbose = verbose
        self.search_stopped = False

    def on_done_other(self, job):
        """Called after gathering local jobs on available remote jobs that are done."""
        self.on_done(job)

    def on_done(self, job):
        """Called when a local job has been gathered."""
        job_objective = self._objective_func(job)

        if self._best_objective is None:
            self._best_objective = job_objective
        else:
            if job_objective > self._best_objective:
                if self._verbose:
                    print(
                        "Objective has improved from "
                        f"{self._best_objective:.5f} -> {job_objective:.5f}"
                    )
                self._best_objective = job_objective
                self._n_lower = 0
            else:
                self._n_lower += 1

        if self._n_lower >= self._patience:
            if self._threshold is None:
                if self._verbose:
                    print(
                        "Stopping the search because it did not improve for the last "
                        f"{self._patience} evaluations!"
                    )
                self.search_stopped = True
            else:
                if self._best_objective > self._threshold:
                    if self._verbose:
                        print(
                            "Stopping the search because it did not improve for the last "
                            f"{self._patience} evaluations!"
                        )
                    self.search_stopped = True


# TODO: Add unit tests
# This class is made to be used by people who wants to log results from the
# evaluator without using it within the Search.
class CSVLoggerCallback(Callback):
    """Dump jobs done to a CSV file.

    Args:
        path (str): The path where the CSV is being dumped.
    """

    def __init__(self, path: str = "results.csv"):
        self.path = os.path.abspath(path)
        if not os.path.exists(os.path.dirname(path)):
            raise ValueError(f"Directory not found {self.path}")
        self.jobs_done = []
        self.num_objective = None
        self._start_dumping = False
        self._columns_dumped = None
        self._job_class = None

    def on_gather(self, local_jobs: List[Job], other_jobs: List[Job]):
        """Called after gathering jobs.

        Args:
            local_jobs (List[Job]):
                gathered jobs from local evaluator instance.

            other_jobs (List[Job]):
                gathered jobs from other evaluators using the same storage.
        """
        self.jobs_done.extend(local_jobs)
        self.jobs_done.extend(other_jobs)
        self.dump_jobs_done_to_csv(self.path)

    def on_close(self):
        self.dump_jobs_done_to_csv(self.path, flush=True)

    def dump_jobs_done_to_csv(self, path: str, flush: bool = False):
        """Dump completed jobs to a CSV file.

        This will reset the ``Evaluator.jobs_done`` attribute to an empty list.

        Args:
            path (str):
                The path of the file where the CSV is being dumped.

            flush (bool):
                A boolean indicating if the results should be flushed (i.e., forcing the dumping).
        """
        if len(self.jobs_done) > 0:
            if self._job_class is None:
                self._job_class = type(self.jobs_done[0])
        else:
            return
        logger.info("Dumping completed jobs to CSV...")
        if self._job_class is HPOJob:
            self._dump_jobs_done_to_csv_as_hpo_format(path, flush)
        else:
            self._dump_jobs_done_to_csv_as_regular_format(path)
        logger.info("Dumping done")

    def _dump_jobs_done_to_csv_as_regular_format(self, path: str):
        """Dump completed jobs to a CSV file for regular job format.

        Args:
            path (str):
                The path of the file where the CSV is being dumped.
        """
        records_list = []

        for job in self.jobs_done:
            # Start with job.id
            result = {"job_id": int(job.id.split(".")[1])}

            # Add job.status
            result["job_status"] = job.status.name

            # input arguments: add prefix for all keys found in "args"
            result.update({f"p:{k}": v for k, v in job.args.items()})

            # output
            if isinstance(job.output, dict):
                output = {f"o:{k}": v for k, v in job.output.items()}
            else:
                output = {"o:": job.output}
            result.update(output)

            # metadata
            metadata = {f"m:{k}": v for k, v in job.metadata.items() if k[0] != "_"}
            result.update(metadata)

            records_list.append(result)

        if len(records_list) != 0:
            mode = "a" if self._start_dumping else "w"

            with open(path, mode) as fp:
                if not (self._start_dumping):
                    self._columns_dumped = records_list[0].keys()

                if self._columns_dumped is not None:
                    writer = csv.DictWriter(fp, self._columns_dumped, extrasaction="ignore")

                    if not (self._start_dumping):
                        writer.writeheader()
                        self._start_dumping = True

                    writer.writerows(records_list)
                    self.jobs_done = []

    def _dump_jobs_done_to_csv_as_hpo_format(self, path: str, flush: bool = False):
        """Dump completed jobs to a CSV file for the hyperparameter optimization format.

        This will reset the ``Evaluator.jobs_done`` attribute to an empty list.

        Args:
            path (str):
                The path of the file where the CSV is being dumped.

            flush (bool):
                A boolean indicating if the results should be flushed (i.e., forcing the dumping).
        """
        resultsList = []

        for job in self.jobs_done:
            # add prefix for all keys found in "args"
            result = {f"p:{k}": v for k, v in job.args.items()}

            # when the returned value of the run-function is a dict we flatten it to add in csv
            result["objective"] = job.objective
            print(f"{job.objective=}")

            # when the objective is a tuple (multi-objective) we create 1 column per tuple-element
            if isinstance(result["objective"], tuple) or isinstance(result["objective"], list):
                obj = result.pop("objective")

                if self.num_objective is None:
                    self.num_objective = len(obj)

                for i, objval in enumerate(obj):
                    result[f"objective_{i}"] = objval
            else:
                if self.num_objective is None:
                    self.num_objective = 1

                if self.num_objective > 1:
                    obj = result.pop("objective")
                    for i in range(self.num_objective):
                        result[f"objective_{i}"] = obj

            # Add job.id
            result["job_id"] = int(job.id.split(".")[1])

            # Add job.status
            result["job_status"] = job.status.name

            # Profiling and other
            # methdata keys starting with "_" are not saved (considered as internal)
            metadata = {f"m:{k}": v for k, v in job.metadata.items() if k[0] != "_"}
            result.update(metadata)

            resultsList.append(result)

        if len(resultsList) != 0:
            mode = "a" if self._start_dumping else "w"

            with open(path, mode) as fp:
                if not (self._start_dumping):
                    for result in resultsList:
                        # Waiting to start receiving non-failed jobs before dumping results
                        is_single_obj_and_has_success = (
                            "objective" in result and type(result["objective"]) is not str
                        )
                        is_multi_obj_and_has_success = (
                            "objective_0" in result and type(result["objective_0"]) is not str
                        )
                        print(f"{is_single_obj_and_has_success=}, {is_multi_obj_and_has_success=}")
                        if is_single_obj_and_has_success or is_multi_obj_and_has_success or flush:
                            self._columns_dumped = result.keys()

                            break

                if self._columns_dumped is not None:
                    writer = csv.DictWriter(fp, self._columns_dumped, extrasaction="ignore")

                    if not (self._start_dumping):
                        writer.writeheader()
                        self._start_dumping = True

                    writer.writerows(resultsList)
                    self.jobs_done = []

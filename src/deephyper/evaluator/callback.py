"""The callback module contains sub-classes of the ``Callback`` class.

The ``Callback`` class is used to trigger custom actions on the start and
completion of jobs by the ``Evaluator``. Callbacks can be used with any
``Evaluator`` implementation.
"""

import abc

import numpy as np

from deephyper.evaluator import HPOJob
from deephyper.evaluator._evaluator import _test_ipython_interpretor
from deephyper.skopt.moo import hypervolume

if _test_ipython_interpretor():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


class Callback(abc.ABC):
    """Callback interface."""

    def on_launch(self, job):
        """Called each time a ``Job`` is created by the ``Evaluator``.

        Args:
            job (Job): The created job.
        """

    def on_done(self, job):
        """Called each time a local ``Job`` has been gathered by the Evaluator.

        Args:
            job (Job): The completed job.
        """

    def on_done_other(self, job):
        """Called after local ``Job`` have been gathered for each remote ``Job`` that is done.

        Args:
            job (Job): The completed Job.
        """


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

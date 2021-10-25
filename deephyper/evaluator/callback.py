"""The callback module contains sub-classes of the ``Callback`` class used to trigger custom actions on the start and completion of jobs by the ``Evaluator``. Callbacks can be used with any Evaluator implementation.
"""
import time

import pandas as pd
from deephyper.core.exceptions import SearchTerminationError


class Callback:
    def on_launch(self, job):
        """Called each time a ``Job`` is created by the ``Evaluator``.

        Args:
            job (Job): The created job.
        """
        ...

    def on_done(self, job):
        """Called each time a Job is completed by the Evaluator.

        Args:
            job (Job): The completed job.
        """
        ...


class ProfilingCallback(Callback):
    """Collect profiling data. Each time a ``Job`` is completed by the ``Evaluator`` a timestamp and current number of running jobs is collected.

    An example usage can be:

    >>> profiler = ProfilingCallback()
    >>> evaluator.create(method="ray", method_kwargs={..., "callbacks": [profiler]})
    ...
    >>> profiler.profile
    """

    def __init__(self):
        self.n = 0
        self.data = []

    def on_launch(self, job):
        t = time.time()
        self.n += 1
        self.data.append([t, self.n])

    def on_done(self, job):
        t = time.time()
        self.n -= 1
        self.data.append([t, self.n])

    @property
    def profile(self):
        cols = ["timestamp", "n_jobs_running"]
        df = pd.DataFrame(self.data, columns=cols)
        return df


class LoggerCallback(Callback):
    """Print information when jobs are completed by the ``Evaluator``.

    An example usage can be:

    >>> evaluator.create(method="ray", method_kwargs={..., "callbacks": [LoggerCallback()]})
    """

    def __init__(self):
        self._best_objective = None
        self._n_done = 0

    def on_done(self, job):
        self._n_done += 1
        if self._best_objective is None:
            self._best_objective = job.result
        else:
            self._best_objective = max(job.result, self._best_objective)
        print(
            f"[{self._n_done:05d}] -- best objective: {self._best_objective:.5f} -- received objective: {job.result:.5f}"
        )


class SearchEarlyStopping(Callback):
    """Stop the search gracefully when it does not improve for a given number of evaluations.

    Args:
        patience (int, optional): The number of not improving evaluations to wait for before stopping the search. Defaults to 10.
        objective_func (callable, optional): A function that takes a ``Job`` has input and returns the maximized scalar value monitored by this callback. Defaults to ``lambda j: j.result``.
    """
    def __init__(self, patience: int = 10, objective_func=lambda j: j.result):
        self._best_objective = None
        self._n_lower = 0
        self._patience = patience
        self._objective_func = objective_func

    def on_done(self, job):
        job_objective = self._objective_func(job)
        if self._best_objective is None:
            self._best_objective = job_objective
        else:
            if job_objective > self._best_objective:
                print(f"Objective has improved from {self._best_objective:.5f} -> {job_objective:.5f}")
                self._best_objective = job_objective
                self._n_lower = 0
            else:
                self._n_lower += 1

        if self._n_lower >= self._patience:
            print(f"Stopping the search because it did not improve for the last {self._patience} evaluations!")
            raise SearchTerminationError

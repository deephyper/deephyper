"""The callback module contains sub-classes of the ``Callback`` class used to trigger custom actions on the start and completion of jobs by the ``Evaluator``. Callbacks can be used with any Evaluator implementation.
"""
import deephyper.core.exceptions
import numpy as np
import pandas as pd
from deephyper.evaluator._evaluator import _test_ipython_interpretor

if _test_ipython_interpretor():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


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
    """Collect profiling data. Each time a ``Job`` is completed by the ``Evaluator`` a the different timestamps corresponding to the submit and gather (and run function start and end if the ``profile`` decorator is used on the run function) are collected.

    An example usage can be:

    >>> profiler = ProfilingCallback()
    >>> evaluator.create(method="ray", method_kwargs={..., "callbacks": [profiler]})
    ...
    >>> profiler.profile
    """

    def __init__(self):
        self.history = []

    def on_launch(self, job):
        ...

    def on_done(self, job):
        start = job.timestamp_submit
        end = job.timestamp_gather
        if job.timestamp_start is not None and job.timestamp_end is not None:
            start = job.timestamp_start
            end = job.timestamp_end
        self.history.append((start, 1))
        self.history.append((end, -1))

    @property
    def profile(self):
        n_jobs = 0
        profile = []
        for t, incr in sorted(self.history):
            n_jobs += incr
            profile.append([t, n_jobs])
        cols = ["timestamp", "n_jobs_running"]
        df = pd.DataFrame(profile, columns=cols)
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
        # Test if multi objectives are received
        if np.ndim(job.result) > 0:
            if np.isreal(job.result).all():
                if self._best_objective is None:
                    self._best_objective = np.sum(job.result)
                else:
                    self._best_objective = max(np.sum(job.result), self._best_objective)

                print(
                    f"[{self._n_done:05d}] -- best sum(objective): {self._best_objective:.5f} -- received sum(objective): {np.sum(job.result):.5f}"
                )
            elif np.any(type(res) is str and "F" == res[0] for res in job.result):
                print(f"[{self._n_done:05d}] -- received failure: {job.result}")
        elif np.isreal(job.result):
            if self._best_objective is None:
                self._best_objective = job.result
            else:
                self._best_objective = max(job.result, self._best_objective)

            print(
                f"[{self._n_done:05d}] -- best objective: {self._best_objective:.5f} -- received objective: {job.result:.5f}"
            )
        elif type(job.result) is str and "F" == job.result[0]:
            print(f"[{self._n_done:05d}] -- received failure: {job.result}")


class TqdmCallback(Callback):
    """Print information when jobs are completed by the ``Evaluator``.

    An example usage can be:

    >>> evaluator.create(method="ray", method_kwargs={..., "callbacks": [TqdmCallback()]})
    """

    def __init__(self):
        self._best_objective = None
        self._n_done = 0
        self._n_failures = 0
        self._max_evals = None
        self._tqdm = None

    def set_max_evals(self, max_evals):
        self._max_evals = max_evals
        self._tqdm = None

    def on_done(self, job):

        if self._tqdm is None:
            if self._max_evals:
                self._tqdm = tqdm(total=self._max_evals)
            else:
                self._tqdm = tqdm()

        self._n_done += 1
        self._tqdm.update(1)
        # Test if multi objectives are received
        if np.ndim(job.result) > 0:
            if np.isreal(job.result).all():
                if self._best_objective is None:
                    self._best_objective = np.sum(job.result)
                else:
                    self._best_objective = max(np.sum(job.result), self._best_objective)
            else:
                self._n_failures += 1
            self._tqdm.set_postfix(
                {"failures": self._n_failures, "sum(objective)": self._best_objective}
            )
        else:
            if np.isreal(job.result):
                if self._best_objective is None:
                    self._best_objective = job.result
                else:
                    self._best_objective = max(job.result, self._best_objective)
            else:
                self._n_failures += 1
            self._tqdm.set_postfix(
                objective=self._best_objective, failures=self._n_failures
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
        # if multi objectives are received
        if np.ndim(job_objective) > 0:
            job_objective = np.sum(job_objective)
        if self._best_objective is None:
            self._best_objective = job_objective
        else:
            if job_objective > self._best_objective:
                print(
                    f"Objective has improved from {self._best_objective:.5f} -> {job_objective:.5f}"
                )
                self._best_objective = job_objective
                self._n_lower = 0
            else:
                self._n_lower += 1

        if self._n_lower >= self._patience:
            print(
                f"Stopping the search because it did not improve for the last {self._patience} evaluations!"
            )
            raise deephyper.core.exceptions.SearchTerminationError

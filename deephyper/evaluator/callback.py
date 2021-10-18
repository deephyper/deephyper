"""The callback module contains sub-classes of the ``Callback`` class used to trigger custom actions on the start and completion of jobs by the ``Evaluator``. Callbacks can be used with any Evaluator implementation.
"""
import pandas as pd
import time


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

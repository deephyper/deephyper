import pandas as pd
import time


class Callback:
    def on_launch(self, job):
        ...

    def on_done(self, job):
        ...


class ProfilingCallback(Callback):
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

    def __init__(self):
        self._best_objective = None
        self._n_done = 0

    def on_done(self, job):
        self._n_done += 1
        if self._best_objective is None:
            self._best_objective = job.result
        else:
            self._best_objective = max(job.result, self._best_objective)
        print(f"[{self._n_done:05d}] -- best objective: {self._best_objective} -- received objective: {job.result}")
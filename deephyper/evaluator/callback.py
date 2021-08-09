
class Callback:

    def on_launch(self, job):
        ...

    def on_done(self, job):
        ...

import pandas as pd
import time

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
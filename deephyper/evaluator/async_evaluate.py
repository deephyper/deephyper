from typing import Dict, List
import asyncio
import os
import sys
import warnings
import csv
import json

import numpy as np
from deephyper.core.exceptions import DeephyperRuntimeError
from deephyper.evaluator.job import Job

class AsyncEvaluator:
    """This evaluator module asynchronously manages a series of Job objects to help execute given HPS or NAS tasks on various environments with differing system settings and properties.

    Raises:
        DeephyperRuntimeError: raised if the `cache_key` parameter is not None, a callable or equal to 'uuid'.
        DeephyperRuntimeError: raised if the `run_function` parameter is from the`__main__` module.

    """

    FAIL_RETURN_VALUE = np.finfo(np.float32).min
    PYTHON_EXE = os.environ.get("DEEPHYPER_PYTHON_BACKEND", sys.executable)
    assert os.path.isfile(PYTHON_EXE)

    def __init__(
        self,
        run_function,
        method,
        num_workers=1,
    ):
        self.run_function = run_function    # User-defined run function.
        self.method = method                # Type of method used
        self.num_workers = num_workers      # Number of processors used for completing some job.
        self.jobs = []                      # Job objects currently submitted.
        self.n_jobs = 1
        self.num_cpus_per_task = 1
        self.num_gpus_per_task = None
        self._tasks_running = []            # List of AsyncIO Task objects currently running.
        self._tasks_done = []               # Temp list to hold completed tasks from asyncio.
        self._tasks_pending = []            # Temp list to hold pending tasks from asyncio.
        self._loop = None                   # Event loop for asyncio.

        moduleName = self.run_function.__module__
        if moduleName == "__main__":
            raise DeephyperRuntimeError(
                f'Evaluator will not execute function " {run_function.__name__}" because it is in the __main__ module.  Please provide a function imported from an external module!'
            )

    async def _get_at_least_n_tasks(self, n):
        # If a user requests a batch size larger than the number of currently-running tasks, set n to the number of tasks running.
        if n > len(self._tasks_running):
            warnings.warn(f"Requested a batch size ({n}) larger than currently running tasks ({len(self._tasks_running)}). Batch size has been set to the count of currently running tasks.")
            n = len(self._tasks_running)

        while len(self._tasks_done) < n:
            self._tasks_done, self._tasks_pending = await asyncio.wait(self._tasks_running,return_when="FIRST_COMPLETED")

    async def _run_jobs(self, configs):
        for config in configs:
            new_job = self.create_job(config)

            task = self.loop.create_task(self.execute(new_job))
            self._tasks_running.append(task)

    def create(
        run_function, 
        method="subprocess", 
        num_workers=1, 
        num_cpus=1, 
        num_gpus=None,
        ray_address=None,
        ray_password=None):

        available_methods = [
            "ray",
            "subprocess",
            "threadPool",
            "processPool"
        ]

        if not method in available_methods:
            raise DeephyperRuntimeError(
                f'The method "{method}" is not a valid method for an Evaluator!'
            )

        if method == "subprocess":
            from deephyper.evaluator._subprocess import SubprocessEvaluator

            Eval = SubprocessEvaluator(run_function, method, num_workers)

        elif method == "threadPool":
            from deephyper.evaluator._threadpool import ThreadPoolEvaluator

            Eval = ThreadPoolEvaluator(run_function, method, num_workers)
        
        elif method == "processPool":
            from deephyper.evaluator._processpool import ProcessPoolEvaluator
            
            Eval = ProcessPoolEvaluator(run_function, method, num_workers) 
        
        elif method == "ray":
            from deephyper.evaluator._ray import RayEvaluator

            Eval = RayEvaluator(
                run_function, 
                method, 
                ray_address,
                ray_password)

        return Eval

    def execute(job):
        raise NotImplementedError

    def submit(self, configs: List[Dict]):
        self.loop = asyncio.get_event_loop()
        self.loop.run_until_complete(self._run_jobs(configs))
        return self.jobs
            

    def gather(self, type, min_batch_size=1):
        results = []

        if type == "ALL":
            min_batch_size = len(self._tasks_running) # Get all tasks.
            self.loop.run_until_complete(self._get_at_least_n_tasks(min_batch_size))
            for task in self._tasks_done:
                job = task.result()
                results.append(job)
                self._tasks_running.remove(task)

        elif type == "BATCH":
            self.loop.run_until_complete(self._get_at_least_n_tasks(min_batch_size))
            for task in self._tasks_done:
                job = task.result()
                results.append(job)
                self._tasks_running.remove(task)

        else:
            raise Exception(f"Unsupported gather operation: {type}.")

        self._tasks_done = []
        self._tasks_pending = []
        return results
    
    def create_job(self, config):
        new_job = Job(
            self.n_jobs, 
            2021, 
            config, 
            self.run_function, 
            self.method, 
            self.num_workers,
            self.num_cpus_per_task,
            self.num_gpus_per_task)
        self.n_jobs += 1 #! @Romain: we can use integers if it is enough, uuid can become useful if jobs'id are generated in parallel
        self.jobs.append(new_job)

        return new_job

    def decode(self, key):
        """from JSON string to x (list)"""
        x = json.loads(key)
        if not isinstance(x, dict):
            raise ValueError(f"Expected dict, but got {type(x)}")
        return x
    
    def dump_evals(self, saved_key: str = None, saved_keys: list = None):
        """Dump evaluations to 'results.csv' file.
        """

        resultsList = []

        for job in self.jobs:
            if job.status is job.DONE and job.printed is False:
                if saved_key is None and saved_keys is None:
                    result = job.config
                elif type(saved_key) is str:
                    result = {str(i): v for i, v in enumerate(job.config[saved_key])}
                elif type(saved_keys) is list:
                    decoded_key = job.config
                    result = {k: self.convert_for_csv(decoded_key[k]) for k in saved_keys}
                elif callable(saved_keys):
                    decoded_key = job.config
                    result = saved_keys(decoded_key)
                result = {"objective": None, "elapsed_sec": None}
                result["objective"] = job.result[1]
                result["elapsed_sec"] = job.duration
                resultsList.append(result)
                job.printed = True

        if len(resultsList) != 0:
            with open("results.csv", "w") as fp:
                columns = resultsList[0].keys()
                writer = csv.DictWriter(fp, columns)
                writer.writeheader()
                writer.writerows(resultsList)







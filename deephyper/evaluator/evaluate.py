import asyncio
import csv
import importlib
import json
import os
import sys
import time
import warnings
from typing import Dict, List

import numpy as np
from deephyper.core.exceptions import DeephyperRuntimeError
from deephyper.evaluator.job import Job

EVALUATORS = {
    "thread": "_thread_pool.ThreadPoolEvaluator",
    "process": "_process_pool.ProcessPoolEvaluator",
    "subprocess": "_subprocess.SubprocessEvaluator",
    "ray": "_ray.RayEvaluator",
    # "balsam": "_balsam.BalsamEvaluator" # TODO
}


class Evaluator:
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
        num_workers=1,
        callbacks=None,
    ):
        self.run_function = run_function  # User-defined run function.
        self.num_workers = (
            num_workers  # Number of processors used for completing some job.
        )
        self.jobs = []  # Job objects currently submitted.
        self.n_jobs = 1
        self.num_cpus_per_task = 1
        self.num_gpus_per_task = None
        self._tasks_running = []  # List of AsyncIO Task objects currently running.
        self._tasks_done = []  # Temp list to hold completed tasks from asyncio.
        self._tasks_pending = []  # Temp list to hold pending tasks from asyncio.
        self.jobs_done = []  # List used to store all jobs completed by the evaluator.
        self.timestamp = (
            time.time()
        )  # Recorded time of when this evaluator interface was created.
        self._loop = None  # Event loop for asyncio.
        self._start_dumping = False

        self._callbacks = [] if callbacks is None else callbacks

    async def _get_at_least_n_tasks(self, n):
        # If a user requests a batch size larger than the number of currently-running tasks, set n to the number of tasks running.
        if n > len(self._tasks_running):
            warnings.warn(
                f"Requested a batch size ({n}) larger than currently running tasks ({len(self._tasks_running)}). Batch size has been set to the count of currently running tasks."
            )
            n = len(self._tasks_running)

        while len(self._tasks_done) < n:
            self._tasks_done, self._tasks_pending = await asyncio.wait(
                self._tasks_running, return_when="FIRST_COMPLETED"
            )

    async def _run_jobs(self, configs):
        for config in configs:
            new_job = self.create_job(config)
            self._on_launch(new_job)
            task = self.loop.create_task(self.execute(new_job))
            self._tasks_running.append(task)

    def create(run_function, method="subprocess", method_kwargs={}):

        if not method in EVALUATORS.keys():
            val = ", ".join(EVALUATORS)
            raise DeephyperRuntimeError(
                f'The method "{method}" is not a valid method for an Evaluator!'
                f' Choose among the following evalutor types: '
                f'{val}.'

            )

        # create the evaluator
        mod_name, attr_name = EVALUATORS[method].split(".")
        mod = importlib.import_module(f"deephyper.evaluator.{mod_name}")
        eval_cls = getattr(mod, attr_name)
        evaluator = eval_cls(run_function, **method_kwargs)

        return evaluator

    def _on_launch(self, job):
        job.status = job.RUNNING

        job.duration = time.time()

        # call callbacks
        for cb in self._callbacks:
            cb.on_launch(job)

    def _on_done(self, job):
        job.status = job.DONE

        job.duration = time.time() - job.duration
        job.elapsed_sec = time.time() - self.timestamp

        # call callbacks
        for cb in self._callbacks:
            cb.on_done(job)

    async def execute(self, job):
        raise NotImplementedError

    def submit(self, configs: List[Dict]):
        self.loop = asyncio.get_event_loop()
        self.loop.run_until_complete(self._run_jobs(configs))
        return self.jobs

    def gather(self, type, size=1):
        """Collect the completed tasks from the evaluator in batches of one or more.

        Args:
            type ([type]):
                Options:
                    "ALL"
                        Collect all jobs submitted to the evaluator.
                        Ex.) eval.gather("ALL")
                    "BATCH"
                        Specify a minimum batch size of jobs to collect from the evaluator.
            size (int, optional): The minimum batch size that we want to collect from the evaluator. Defaults to 1.

        Raises:
            Exception: Raised when a gather operation other than "ALL" or "BATCH" is provided.

        Returns:
            List[Job]: A batch of completed jobs that is at minimum the given size.
        """
        assert type in ["ALL", "BATCH"], f"Unsupported gather operation: {type}."

        results = []

        if type == "ALL":
            size = len(self._tasks_running)  # Get all tasks.

        self.loop.run_until_complete(self._get_at_least_n_tasks(size))
        for task in self._tasks_done:
            job = task.result()
            self._on_done(job)
            results.append(job)
            self.jobs_done.append(job)
            self._tasks_running.remove(task)

        self._tasks_done = []
        self._tasks_pending = []
        return results

    def create_job(self, config):
        new_job = Job(
            self.n_jobs,
            2021,
            config,
            self.run_function,
            self.num_workers,
            self.num_cpus_per_task,
            self.num_gpus_per_task,
        )
        self.n_jobs += 1  #! @Romain: we can use integers if it is enough, uuid can become useful if jobs'id are generated in parallel
        self.jobs.append(new_job)

        return new_job

    def decode(self, key):
        """from JSON string to x (list)"""
        x = json.loads(key)
        if not isinstance(x, dict):
            raise ValueError(f"Expected dict, but got {type(x)}")
        return x

    def dump_evals(self, saved_key: str = None, saved_keys: list = None):
        """Dump evaluations to 'results.csv' file."""

        resultsList = []

        for job in self.jobs_done:
            if saved_key is None and saved_keys is None:
                result = job.config
            elif type(saved_key) is str:
                result = {str(i): v for i, v in enumerate(job.config[saved_key])}
            elif type(saved_keys) is list:
                decoded_key = job.config
                result = {k: self.convert_for_csv(decoded_key[k]) for k in saved_keys}
            elif callable(saved_keys):
                result = saved_keys(job)
            result["id"] = job.id
            result["objective"] = job.result
            result[
                "elapsed_sec"
            ] = job.elapsed_sec  # Time to complete from the intitilization of evaluator.
            result["duration"] = job.duration
            resultsList.append(result)

            self.jobs_done.remove(job)

        if len(resultsList) != 0:
            mode = "a" if self._start_dumping else "w"
            with open("results.csv", mode) as fp:
                columns = resultsList[0].keys()
                writer = csv.DictWriter(fp, columns)
                if not(self._start_dumping):
                    writer.writeheader()
                    self._start_dumping = True
                writer.writerows(resultsList)

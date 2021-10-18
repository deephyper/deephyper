import asyncio
import csv
import copy
import importlib
import json
import os
import sys
import time
import warnings
from typing import Dict, List

import numpy as np
from deephyper.core.exceptions import DeephyperRuntimeError
from deephyper.evaluator._job import Job

EVALUATORS = {
    "thread": "_thread_pool.ThreadPoolEvaluator",
    "process": "_process_pool.ProcessPoolEvaluator",
    "subprocess": "_subprocess.SubprocessEvaluator",
    "ray": "_ray.RayEvaluator",
    # "balsam": "_balsam.BalsamEvaluator" # TODO
}


class Evaluator:
    """This ``Evaluator`` class asynchronously manages a series of Job objects to help execute given HPS or NAS tasks on various environments with differing system settings and properties.

    Args:
        run_function (callable): functions to be executed by the ``Evaluator``.
        num_workers (int, optional): Number of parallel workers available for the ``Evaluator``. Defaults to 1.
        callbacks (list, optional): A list of callbacks to trigger custom actions at the creation or completion of jobs. Defaults to None.
    """

    FAIL_RETURN_VALUE = np.finfo(np.float32).min
    PYTHON_EXE = os.environ.get("DEEPHYPER_PYTHON_BACKEND", sys.executable)
    assert os.path.isfile(PYTHON_EXE)

    def __init__(
        self,
        run_function,
        num_workers: int = 1,
        callbacks=None,
    ):
        self.run_function = run_function  # User-defined run function.

        # Number of parallel workers available
        self.num_workers =  num_workers
        self.jobs = []  # Job objects currently submitted.
        self.n_jobs = 1
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

    @staticmethod
    def create(run_function, method="subprocess", method_kwargs={}):
        """Create evaluator with a specific backend and configuration.

        Args:
            run_function (function): the function to execute in parallel.
            method (str, optional): the backend to use in ["thread", "process", "subprocess", "ray"]. Defaults to "subprocess".
            method_kwargs (dict, optional): configuration dictionnary of the corresponding backend. Keys corresponds to the keyword arguments of the corresponding implementation. Defaults to "{}".

        Raises:
            DeephyperRuntimeError: if the ``method is`` not acceptable.

        Returns:
            Evaluator: the ``Evaluator`` with the corresponding backend and configuration.
        """

        if not method in EVALUATORS.keys():
            val = ", ".join(EVALUATORS)
            raise DeephyperRuntimeError(
                f'The method "{method}" is not a valid method for an Evaluator!'
                f" Choose among the following evalutor types: "
                f"{val}."
            )

        # create the evaluator
        mod_name, attr_name = EVALUATORS[method].split(".")
        mod = importlib.import_module(f"deephyper.evaluator.{mod_name}")
        eval_cls = getattr(mod, attr_name)
        evaluator = eval_cls(run_function, **method_kwargs)

        return evaluator

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

    def _on_launch(self, job):
        """Called after a job is started."""
        job.status = job.RUNNING

        job.duration = time.time()

        # call callbacks
        for cb in self._callbacks:
            cb.on_launch(job)

    def _on_done(self, job):
        """Called after a job has completed."""
        job.status = job.DONE

        job.duration = time.time() - job.duration
        job.elapsed_sec = time.time() - self.timestamp

        if np.isscalar(job.result):
            if not (np.isfinite(job.result)):
                job.result = Evaluator.FAIL_RETURN_VALUE

        # call callbacks
        for cb in self._callbacks:
            cb.on_done(job)

    async def execute(self, job):
        """Execute the received job. To be implemented with a specific backend.

        Args:
            job (Job): the ``Job`` to be executed.
        """
        raise NotImplementedError

    def submit(self, configs: List[Dict]):
        """Send configurations to be evaluated by available workers.

        Args:
            configs (List[Dict]): A list of dict which will be passed to the run function to be executed.
        """
        self.loop = asyncio.get_event_loop()
        self.loop.run_until_complete(self._run_jobs(configs))

    def gather(self, type, size=1):
        """Collect the completed tasks from the evaluator in batches of one or more.

        Args:
            type (str):
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
        new_job = Job(self.n_jobs, config, self.run_function)
        self.n_jobs += 1
        self.jobs.append(new_job)

        return new_job

    def decode(self, key):
        """from JSON string to x (list)"""
        x = json.loads(key)
        if not isinstance(x, dict):
            raise ValueError(f"Expected dict, but got {type(x)}")
        return x

    def convert_for_csv(self, val):
        """Convert an input value to an accepted format to be saved as a value of a CSV file (e.g., a list becomes it's str representation).

        Args:
            val (Any): The input value to convert.

        Returns:
            Any: The converted value.
        """
        if type(val) is list:
            return str(val)
        else:
            return val

    def dump_evals(
        self, saved_keys = None, log_dir: str = "."
    ):
        """Dump evaluations to a CSV file name ``"results.csv"``

        Args:
            saved_keys (list|callable): If ``None`` the whole ``job.config`` will be added as row of the CSV file. If a ``list`` filtered keys will be added as a row of the CSV file. If a ``callable`` the output dictionnary will be added as a row of the CSV file.
            log_dir (str): directory where to dump the CSV file.
        """

        resultsList = []

        for job in self.jobs_done:
            if saved_keys is None:
                result = copy.deepcopy(job.config)
            elif type(saved_keys) is list:
                decoded_key = copy.deepcopy(job.config)
                result = {k: self.convert_for_csv(decoded_key[k]) for k in saved_keys}
            elif callable(saved_keys):
                result = copy.deepcopy(saved_keys(job))
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
            with open(os.path.join(log_dir, "results.csv"), mode) as fp:
                columns = resultsList[0].keys()
                writer = csv.DictWriter(fp, columns)
                if not (self._start_dumping):
                    writer.writeheader()
                    self._start_dumping = True
                writer.writerows(resultsList)

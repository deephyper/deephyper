import asyncio
import copy
import csv
import importlib
import json
import logging
import os
import sys
import time
import warnings
from typing import Dict, List

import numpy as np
from deephyper.evaluator._job import Job
from deephyper.skopt.optimizer import OBJECTIVE_VALUE_FAILURE
from deephyper.core.utils._introspection import get_init_params_as_json

EVALUATORS = {
    "mpipool": "_mpi_pool.MPIPoolEvaluator",
    "mpicomm": "_mpi_comm.MPICommEvaluator",
    "process": "_process_pool.ProcessPoolEvaluator",
    "ray": "_ray.RayEvaluator",
    "serial": "_serial.SerialEvaluator",
    "subprocess": "_subprocess.SubprocessEvaluator",
    "thread": "_thread_pool.ThreadPoolEvaluator",
}


def _test_ipython_interpretor() -> bool:
    """Test if the current Python interpretor is IPython or not.

    Suggested by: https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook
    """

    # names of shells/modules using jupyter
    notebooks_shells = ["ZMQInteractiveShell"]
    notebooks_modules = ["google.colab._shell"]

    try:
        shell_name = get_ipython().__class__.__name__  # type: ignore
        shell_module = get_ipython().__class__.__module__  # type: ignore

        if shell_name in notebooks_shells or shell_module in notebooks_modules:
            return True  # Jupyter notebook or qtconsole
        elif shell_name == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)

    except NameError:
        return False  # Probably standard Python interpreter


class Evaluator:
    """This ``Evaluator`` class asynchronously manages a series of Job objects to help execute given HPS or NAS tasks on various environments with differing system settings and properties.

    Args:
        run_function (callable): functions to be executed by the ``Evaluator``.
        num_workers (int, optional): Number of parallel workers available for the ``Evaluator``. Defaults to 1.
        callbacks (list, optional): A list of callbacks to trigger custom actions at the creation or completion of jobs. Defaults to None.
    """

    FAIL_RETURN_VALUE = OBJECTIVE_VALUE_FAILURE
    NEST_ASYNCIO_PATCHED = False
    PYTHON_EXE = os.environ.get("DEEPHYPER_PYTHON_BACKEND", sys.executable)
    assert os.path.isfile(PYTHON_EXE)

    def __init__(
        self,
        run_function,
        num_workers: int = 1,
        callbacks: list = None,
        run_function_kwargs: dict = None,
    ):
        self.run_function = run_function  # User-defined run function.
        self.run_function_kwargs = (
            {} if run_function_kwargs is None else run_function_kwargs
        )

        # Number of parallel workers available
        self.num_workers = num_workers
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
        self.num_objective = None  # record if multi-objective are recorded

        self._callbacks = [] if callbacks is None else callbacks

        self._lock = asyncio.Lock()

        # to avoid "RuntimeError: This event loop is already running"
        if not (Evaluator.NEST_ASYNCIO_PATCHED) and _test_ipython_interpretor():
            warnings.warn(
                "Applying nest-asyncio patch for IPython Shell!", category=UserWarning
            )
            import deephyper.evaluator._nest_asyncio as nest_asyncio

            nest_asyncio.apply()
            Evaluator.NEST_ASYNCIO_PATCHED = True

    def to_json(self):
        """Returns a json version of the evaluator."""
        out = {"type": type(self).__name__, **get_init_params_as_json(self)}
        return out

    @staticmethod
    def create(run_function, method="serial", method_kwargs={}):
        """Create evaluator with a specific backend and configuration.

        Args:
            run_function (function): the function to execute in parallel.
            method (str, optional): the backend to use in ``["serial", "thread", "process", "subprocess", "ray", "mpicomm", "mpipool"]``. Defaults to ``"serial"``.
            method_kwargs (dict, optional): configuration dictionnary of the corresponding backend. Keys corresponds to the keyword arguments of the corresponding implementation. Defaults to "{}".

        Raises:
            ValueError: if the ``method is`` not acceptable.

        Returns:
            Evaluator: the ``Evaluator`` with the corresponding backend and configuration.
        """
        logging.info(
            f"Creating Evaluator({run_function}, method={method}, method_kwargs={method_kwargs}..."
        )
        if not method in EVALUATORS.keys():
            val = ", ".join(EVALUATORS)
            raise ValueError(
                f'The method "{method}" is not a valid method for an Evaluator!'
                f" Choose among the following evalutor types: "
                f"{val}."
            )

        # create the evaluator
        mod_name, attr_name = EVALUATORS[method].split(".")
        mod = importlib.import_module(f"deephyper.evaluator.{mod_name}")
        eval_cls = getattr(mod, attr_name)
        evaluator = eval_cls(run_function, **method_kwargs)

        logging.info(f"Creation done")

        return evaluator

    async def _get_at_least_n_tasks(self, n):
        # If a user requests a batch size larger than the number of currently-running tasks, set n to the number of tasks running.
        if n > len(self._tasks_running):
            warnings.warn(
                f"Requested a batch size ({n}) larger than currently running tasks ({len(self._tasks_running)}). Batch size has been set to the count of currently running tasks."
            )
            n = len(self._tasks_running)

        # wait for all running tasks (sync.)
        if n == len(self._tasks_running):
            self._tasks_done, self._tasks_pending = await asyncio.wait(
                self._tasks_running, return_when="ALL_COMPLETED"
            )
        else:
            while len(self._tasks_done) < n:
                self._tasks_done, self._tasks_pending = await asyncio.wait(
                    self._tasks_running, return_when="FIRST_COMPLETED"
                )

    async def _run_jobs(self, configs):
        for config in configs:

            # Create a Job object from the input configuration
            new_job = Job(self.n_jobs, config, self.run_function)
            self.n_jobs += 1
            self.jobs.append(new_job)

            self._on_launch(new_job)
            task = self.loop.create_task(self._execute(new_job))
            self._tasks_running.append(task)

    def _on_launch(self, job):
        """Called after a job is started."""
        job.status = job.RUNNING

        job.timestamp_submit = time.time() - self.timestamp

        # call callbacks
        for cb in self._callbacks:
            cb.on_launch(job)

    def _on_done(self, job):
        """Called after a job has completed."""
        job.status = job.DONE

        job.timestamp_gather = time.time() - self.timestamp

        if np.isscalar(job.result):
            if np.isreal(job.result) and not (np.isfinite(job.result)):
                job.result = Evaluator.FAIL_RETURN_VALUE

        # call callbacks
        for cb in self._callbacks:
            cb.on_done(job)

    async def _execute(self, job):

        job = await self.execute(job)

        # code to manage the profile decorator
        profile_keys = ["objective", "timestamp_start", "timestamp_end"]
        if isinstance(job.result, dict) and all(k in job.result for k in profile_keys):
            profile = job.result
            job.result = profile["objective"]
            job.timestamp_start = profile["timestamp_start"] - self.timestamp
            job.timestamp_end = profile["timestamp_end"] - self.timestamp

        return job

    async def execute(self, job) -> Job:
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
        logging.info(f"submit {len(configs)} job(s) starts...")
        self.loop = asyncio.get_event_loop()
        self.loop.run_until_complete(self._run_jobs(configs))
        logging.info("submit done")

    def gather(self, type, size=1):
        """Collect the completed tasks from the evaluator in batches of one or more.

        Args:
            type (str):
                Options:
                    ``"ALL"``
                        Block until all jobs submitted to the evaluator are completed.
                    ``"BATCH"``
                        Specify a minimum batch size of jobs to collect from the evaluator. The method will block until at least ``size`` evaluations are completed.
            size (int, optional): The minimum batch size that we want to collect from the evaluator. Defaults to 1.

        Raises:
            Exception: Raised when a gather operation other than "ALL" or "BATCH" is provided.

        Returns:
            List[Job]: A batch of completed jobs that is at minimum the given size.
        """
        logging.info(f"gather({type}, size={size}) starts...")
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
        logging.info("gather done")
        return results

    def decode(self, key):
        """Decode the key following a JSON format to return a dict."""
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

    def dump_evals(self, saved_keys=None, log_dir: str = ".", filename="results.csv"):
        """Dump evaluations to a CSV file.

        Args:
            saved_keys (list|callable): If ``None`` the whole ``job.config`` will be added as row of the CSV file. If a ``list`` filtered keys will be added as a row of the CSV file. If a ``callable`` the output dictionnary will be added as a row of the CSV file.
            log_dir (str): directory where to dump the CSV file.
            filename (str): name of the file where to write the data.
        """
        logging.info("dump_evals starts...")
        resultsList = []

        for job in self.jobs_done:

            if saved_keys is None:
                result = copy.deepcopy(job.config)
            elif type(saved_keys) is list:
                decoded_key = copy.deepcopy(job.config)
                result = {k: self.convert_for_csv(decoded_key[k]) for k in saved_keys}
            elif callable(saved_keys):
                result = copy.deepcopy(saved_keys(job))

            result["job_id"] = job.id

            # when the returned value of the bb is a dict we flatten it to add in csv
            if isinstance(job.result, dict):
                result.update(job.result)
            else:
                result["objective"] = job.result

            # when the objective is a tuple we create 1 column per tuple-element
            if isinstance(result["objective"], tuple):
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

            result["timestamp_submit"] = job.timestamp_submit
            result["timestamp_gather"] = job.timestamp_gather

            if job.timestamp_start is not None and job.timestamp_end is not None:
                result["timestamp_start"] = job.timestamp_start
                result["timestamp_end"] = job.timestamp_end

            if hasattr(job, "dequed"):
                result["dequed"] = ",".join(job.dequed)

            if "optuna_trial" in result:
                result.pop("optuna_trial")

            resultsList.append(result)

        self.jobs_done = []

        if len(resultsList) != 0:
            mode = "a" if self._start_dumping else "w"
            with open(os.path.join(log_dir, filename), mode) as fp:
                columns = resultsList[0].keys()
                writer = csv.DictWriter(fp, columns)
                if not (self._start_dumping):
                    writer.writeheader()
                    self._start_dumping = True
                writer.writerows(resultsList)

        logging.info("dump_evals done")

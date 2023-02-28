import asyncio
import copy
import csv
import functools
import importlib
import json
import logging
import os
import sys
import time
import warnings
from typing import Dict, List, Hashable

import numpy as np
from deephyper.evaluator._job import Job
from deephyper.skopt.optimizer import OBJECTIVE_VALUE_FAILURE
from deephyper.core.utils._timeout import terminate_on_timeout
from deephyper.evaluator.storage import Storage, MemoryStorage

EVALUATORS = {
    "mpicomm": "_mpi_comm.MPICommEvaluator",
    "process": "_process_pool.ProcessPoolEvaluator",
    "ray": "_ray.RayEvaluator",
    "serial": "_serial.SerialEvaluator",
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
        run_function_kwargs (dict, optional): Static keyword arguments to pass to the ``run_function`` when executed.
        storage (Storage, optional): Storage used by the evaluator. Defaults to ``MemoryStorage``.
        search_id (Hashable, optional): The id of the search to use in the corresponding storage. If ``None`` it will create a new search identifier when initializing the search.
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
        storage: Storage = None,
        search_id: Hashable = None,
    ):
        self.run_function = run_function  # User-defined run function.
        self.run_function_kwargs = (
            {} if run_function_kwargs is None else run_function_kwargs
        )

        # Number of parallel workers available
        self.num_workers = num_workers
        self.jobs = []  # Job objects currently submitted.
        self._tasks_running = []  # List of AsyncIO Task objects currently running.
        self._tasks_done = []  # Temp list to hold completed tasks from asyncio.
        self._tasks_pending = []  # Temp list to hold pending tasks from asyncio.
        self.jobs_done = []  # List used to store all jobs completed by the evaluator.
        self.job_id_gathered = []  # List of jobs'id gathered by the evaluator.
        self.timestamp = (
            time.time()
        )  # Recorded time of when this evaluator interface was created.
        self.loop = None  # Event loop for asyncio.
        self._start_dumping = False
        self.num_objective = None  # record if multi-objective are recorded
        self._stopper = None  # stopper object

        self._callbacks = [] if callbacks is None else callbacks

        self._lock = asyncio.Lock()

        # manage timeout of the search
        self._time_timeout_set = None
        self._timeout = None

        # storage mechanism
        self._storage = MemoryStorage() if storage is None else storage
        if not (self._storage.connected):
            self._storage.connect()

        if search_id is None:
            self._search_id = self._storage.create_new_search()
        else:
            if search_id in self._storage.load_all_search_ids():
                self._search_id = search_id
            else:
                raise ValueError(
                    f"The given search_id={search_id} does not exist in the linked storage."
                )

        # to avoid "RuntimeError: This event loop is already running"
        if not (Evaluator.NEST_ASYNCIO_PATCHED) and _test_ipython_interpretor():
            warnings.warn(
                "Applying nest-asyncio patch for IPython Shell!", category=UserWarning
            )
            import deephyper.evaluator._nest_asyncio as nest_asyncio

            nest_asyncio.apply()
            Evaluator.NEST_ASYNCIO_PATCHED = True

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        if hasattr(self, "executor"):
            self.executor.__exit__(type, value, traceback)

    def set_timeout(self, timeout):
        """Set a timeout for the Evaluator. It will create task with a "time budget" and will kill the the task if this budget
        is exhausted."""
        self._time_timeout_set = time.time()
        self._timeout = timeout

    def to_json(self):
        """Returns a json version of the evaluator."""
        out = {"type": type(self).__name__, "num_workers": self.num_workers}
        return out

    @staticmethod
    def create(run_function, method="serial", method_kwargs={}):
        """Create evaluator with a specific backend and configuration.

        Args:
            run_function (function): the function to execute in parallel.
            method (str, optional): the backend to use in ``["serial", "thread", "process", "ray", "mpicomm"]``. Defaults to ``"serial"``.
            method_kwargs (dict, optional): configuration dictionnary of the corresponding backend. Keys corresponds to the keyword arguments of the corresponding implementation. Defaults to "{}".

        Raises:
            ValueError: if the ``method is`` not acceptable.

        Returns:
            Evaluator: the ``Evaluator`` with the corresponding backend and configuration.
        """
        logging.info(
            f"Creating Evaluator({run_function}, method={method}, method_kwargs={method_kwargs}..."
        )
        if method not in EVALUATORS.keys():
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

        logging.info("Creation done")

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
            try:
                self._tasks_done, self._tasks_pending = await asyncio.wait(
                    self._tasks_running, return_when="ALL_COMPLETED"
                )
            except ValueError:
                raise ValueError("No jobs pending, call Evaluator.submit(jobs)!")
        else:
            while len(self._tasks_done) < n:
                self._tasks_done, self._tasks_pending = await asyncio.wait(
                    self._tasks_running, return_when="FIRST_COMPLETED"
                )

    async def _run_jobs(self, configs):
        for config in configs:

            # Create a Job object from the input configuration
            job_id = self._storage.create_new_job(self._search_id)
            self._storage.store_job_in(job_id, args=(config,))
            new_job = Job(job_id, config, self.run_function)

            if self._timeout:
                time_consumed = time.time() - self._time_timeout_set
                time_left = self._timeout - time_consumed
                logging.info(f"Submitting job with {time_left} sec. time budget")
                new_job.run_function = functools.partial(
                    terminate_on_timeout, time_left, new_job.run_function
                )

            self.jobs.append(new_job)

            self._on_launch(new_job)
            task = self.loop.create_task(self._execute(new_job))
            self._tasks_running.append(task)

    def _on_launch(self, job):
        """Called after a job is started."""
        job.status = job.RUNNING

        job.output["metadata"]["timestamp_submit"] = time.time() - self.timestamp

        # call callbacks
        for cb in self._callbacks:
            cb.on_launch(job)

    def _on_done(self, job):
        """Called after a job has completed."""
        job.status = job.DONE

        job.output["metadata"]["timestamp_gather"] = time.time() - self.timestamp

        if np.isscalar(job.objective):
            if np.isreal(job.objective) and not (np.isfinite(job.objective)):
                job.output["objective"] = Evaluator.FAIL_RETURN_VALUE

        # store data in storage
        self._storage.store_job_out(job.id, job.objective)
        for k, v in job.metadata.items():
            self._storage.store_job_metadata(job.id, k, v)

        # call callbacks
        for cb in self._callbacks:
            cb.on_done(job)

    async def _execute(self, job):

        job = await self.execute(job)

        if not (isinstance(job.output, dict)):
            raise ValueError(
                "The output of the job is not standard. Check if `job.set_output(output) was correctly used when defining the Evaluator class."
            )

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
        if self.loop is None:
            try:
                # works if `timeout` is not set and code is running in main thread
                self.loop = asyncio.get_event_loop()
            except RuntimeError:
                # required when `timeout` is set because code is not running in main thread
                self.loop = asyncio.new_event_loop()
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

        local_results = []

        if type == "ALL":
            size = len(self._tasks_running)  # Get all tasks.

        self.loop.run_until_complete(self._get_at_least_n_tasks(size))
        for task in self._tasks_done:
            job = task.result()
            self._on_done(job)
            local_results.append(job)
            self.jobs_done.append(job)
            self._tasks_running.remove(task)
            self.job_id_gathered.append(job.id)

        self._tasks_done = []
        self._tasks_pending = []

        # access storage to return results from other processes
        job_id_all = self._storage.load_all_job_ids(self._search_id)
        job_id_not_gathered = np.setdiff1d(job_id_all, self.job_id_gathered).tolist()

        other_results = []
        if len(job_id_not_gathered) > 0:
            jobs_data = self._storage.load_jobs(job_id_not_gathered)

            for job_id in job_id_not_gathered:
                job_data = jobs_data[job_id]
                if job_data and job_data["out"]:
                    job = Job(
                        id=job_id, config=job_data["in"]["args"][0], run_function=None
                    )
                    job.status = Job.DONE
                    job.output["metadata"].update(job_data["metadata"])
                    job.output["objective"] = job_data["out"]
                    self.job_id_gathered.append(job_id)
                    self.jobs_done.append(job)
                    other_results.append(job)

                    for cb in self._callbacks:
                        cb.on_done_other(job)

        if len(other_results) == 0:
            logging.info(f"gather done - {len(local_results)} job(s)")

            return local_results
        else:
            logging.info(
                f"gather done - {len(local_results)} local(s) and {len(other_results)} other(s) job(s), "
            )

            return local_results, other_results

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

            # add prefix for all keys found in "config"
            result = {f"p:{k}": v for k, v in result.items()}

            # when the returned value of the run-function is a dict we flatten it to add in csv
            result["objective"] = job.objective

            # when the objective is a tuple (multi-objective) we create 1 column per tuple-element
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

            # job id and rank
            result["job_id"] = int(job.id.split(".")[1])

            if isinstance(job.rank, int):
                result["rank"] = job.rank

            # Profiling and other
            # methdata keys starting with "_" are not saved (considered as internal)
            metadata = {f"m:{k}": v for k, v in job.metadata.items() if k[0] != "_"}
            result.update(metadata)

            if hasattr(job, "dequed"):
                result["m:dequed"] = ",".join(job.dequed)

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

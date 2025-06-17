import abc
import asyncio
import importlib
import json
import logging
import os
import sys
import time
import warnings
from typing import Dict, Hashable, List, Optional

import numpy as np

from deephyper.evaluator._job import HPOJob, Job, JobStatus
from deephyper.evaluator.storage import MemoryStorage, Storage
from deephyper.evaluator.utils import test_ipython_interpretor
from deephyper.skopt.optimizer import OBJECTIVE_VALUE_FAILURE

EVALUATORS = {
    "mpicomm": "_mpi_comm.MPICommEvaluator",
    "process": "_process_pool.ProcessPoolEvaluator",
    "ray": "_ray.RayEvaluator",
    "serial": "_serial.SerialEvaluator",
    "thread": "_thread_pool.ThreadPoolEvaluator",
    "loky": "_loky.LokyEvaluator",
}


class MaximumJobsSpawnReached(RuntimeError):
    """Raised when the maximum number of jobs spawned by the evaluator is reached."""


class Evaluator(abc.ABC):
    """This ``Evaluator`` class asynchronously manages a series of Job objects.

    It helps to execute given HPS or NAS tasks on various environments with
    differing system settings and properties.

    Args:
        run_function (callable):
            Functions to be executed by the ``Evaluator``.
        num_workers (int, optional):
            Number of parallel workers available for the ``Evaluator``. Defaults to 1.
        callbacks (list, optional):
            A list of callbacks to trigger custom actions at the creation or
            completion of jobs. Defaults to None.
        run_function_kwargs (dict, optional):
            Static keyword arguments to pass to the ``run_function`` when executed.
        storage (Storage, optional):
            Storage used by the evaluator. Defaults to ``MemoryStorage``.
        search_id (Hashable, optional):
            The id of the search to use in the corresponding storage. If
            ``None`` it will create a new search identifier when initializing
            the search.
    """

    FAIL_RETURN_VALUE = OBJECTIVE_VALUE_FAILURE
    NEST_ASYNCIO_PATCHED = False
    PYTHON_EXE = os.environ.get("DEEPHYPER_PYTHON_BACKEND", sys.executable)
    assert os.path.isfile(PYTHON_EXE)

    def __init__(
        self,
        run_function,
        num_workers: Optional[int] = 1,
        callbacks: Optional[list] = None,
        run_function_kwargs: Optional[dict] = None,
        storage: Optional[Storage] = None,
        search_id: Optional[Hashable] = None,
    ):
        if hasattr(run_function, "__name__") and hasattr(run_function, "__module__"):
            logging.info(
                f"{type(self).__name__} will execute {run_function.__name__}() from module "
                f"{run_function.__module__}"
            )
        else:
            logging.info(f"{type(self).__name__} will execute {run_function}")

        self.run_function = run_function  # User-defined run function.
        self.run_function_kwargs = {} if run_function_kwargs is None else run_function_kwargs

        # Number of parallel workers available
        self.num_workers = num_workers
        self.jobs = []  # Job objects currently submitted.
        self._tasks_running = []  # List of AsyncIO Task objects currently running.
        self._tasks_done = []  # Temp list to hold completed tasks from asyncio.
        self._tasks_pending = []  # Temp list to hold pending tasks from asyncio.
        self.job_id_submitted = []  # List of jobs'id submitted by the evaluator.
        self.job_id_gathered = []  # List of jobs'id gathered by the evaluator.
        self.timestamp = time.time()  # Recorded time of when this evaluator interface was created.
        self.maximum_num_jobs_submitted = -1  # Maximum number of jobs to spawn.
        self._num_jobs_offset = 0
        self.loop: Optional[asyncio.AbstractEventLoop] = None  # Event loop for asyncio.
        self.num_objective = None  # record if multi-objective are recorded
        self._stopper = None  # stopper object
        self.search = None  # search instance

        self._callbacks = [] if callbacks is None else callbacks

        self._lock = asyncio.Lock()

        # manage timeout of the search
        self._time_timeout_set = None
        self._timeout = None

        # storage mechanism
        self._storage = MemoryStorage() if storage is None else storage
        if not self._storage.is_connected():
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
        if not (Evaluator.NEST_ASYNCIO_PATCHED) and test_ipython_interpretor():
            warnings.warn("Applying nest-asyncio patch for IPython Shell!", category=UserWarning)
            import deephyper.evaluator._nest_asyncio as nest_asyncio

            nest_asyncio.apply()
            Evaluator.NEST_ASYNCIO_PATCHED = True

        self._job_class = Job

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        if hasattr(self, "executor"):
            self.executor.__exit__(type, value, traceback)  # type: ignore

    @property
    def timeout(self):
        return self._timeout

    @timeout.setter
    def timeout(self, value):
        """Set a timeout for the Evaluator.

        It will create new tasks with a "time budget" and it will cancel the
        the task if this budget is exhausted.
        """
        self._time_timeout_set = time.time()
        self._timeout = value

    @property
    def time_left(self):
        """Returns the time remaining according to a previously set timeout."""
        if self.timeout is None:
            val = None
        else:
            if self._time_timeout_set is not None:
                time_consumed = time.time() - self._time_timeout_set
                val = self.timeout - time_consumed
            else:
                raise RuntimeError(f"{self._time_timeout_set=} should not be set to a float")
        logging.info(f"time_left={val}")
        return val

    def set_maximum_num_jobs_submitted(self, maximum_num_jobs_submitted: int):
        # TODO: use storage to count submitted and gathered jobs...
        # TODO: should be a property with a setter?
        self.maximum_num_jobs_submitted = maximum_num_jobs_submitted
        self._num_jobs_offset = self.num_jobs_gathered

    @property
    def num_jobs_submitted(self):
        job_ids = self._storage.load_all_job_ids(self._search_id)
        return len(job_ids) - self._num_jobs_offset

    @property
    def num_jobs_gathered(self):
        return len(self.job_id_gathered) - self._num_jobs_offset

    def to_json(self):
        """Returns a json version of the evaluator."""
        out = {"type": type(self).__name__, "num_workers": self.num_workers}
        return out

    @staticmethod
    def create(run_function, method="serial", method_kwargs={}):
        """Create evaluator with a specific backend and configuration.

        Args:
            run_function (function):
                The function to execute in parallel.
            method (str, optional):
                The backend to use in ``
                ["serial", "thread", "process", "ray", "mpicomm"]``. Defaults
                to ``"serial"``.
            method_kwargs (dict, optional):
                Configuration dictionnary of the corresponding backend. Keys
                corresponds to the keyword arguments of the corresponding
                implementation. Defaults to "{}".

        Raises:
            ValueError: if the ``method is`` not acceptable.

        Returns:
            Evaluator: the ``Evaluator`` with the corresponding backend and configuration.
        """
        if method not in EVALUATORS.keys():
            val = ", ".join(EVALUATORS)
            raise ValueError(
                f'The method "{method}" is not a valid method for an Evaluator!'
                f" Choose among the following evalutor types: "
                f"{val}."
            )

        logging.info(
            f"Creating {EVALUATORS[method].split('.')[-1]} of {method=} for "
            f"{run_function=} with {method_kwargs=}"
        )

        # create the evaluator
        mod_name, attr_name = EVALUATORS[method].split(".")
        mod = importlib.import_module(f"deephyper.evaluator.{mod_name}")
        eval_cls = getattr(mod, attr_name)
        evaluator = eval_cls(run_function, **method_kwargs)

        logging.info("Creation done")

        return evaluator

    def _create_job(self, job_id, args, run_function, storage) -> Job:
        return self._job_class(job_id, args, run_function, storage)

    async def _await_at_least_n_tasks(self, n):
        # If a user requests a batch size larger than the number of
        # currently-running tasks, set n to the number of tasks running.
        if n > len(self._tasks_running):
            warnings.warn(
                "Requested a batch size ({n}) larger than currently running tasks "
                f"({len(self._tasks_running)}). Batch size has been set to the count of currently "
                "running tasks."
            )
            n = len(self._tasks_running)

        # wait for all running tasks (sync.)
        if n == len(self._tasks_running):
            try:
                self._tasks_done, self._tasks_pending = await asyncio.wait(
                    self._tasks_running,
                    return_when="ALL_COMPLETED",
                )
            except asyncio.CancelledError:
                logging.warning("Cancelled running tasks")
                self._tasks_done = []
                self._tasks_pending = []
            except ValueError:
                raise ValueError("No jobs pending, call Evaluator.submit(jobs)!")
        else:
            while len(self._tasks_done) < n:
                self._tasks_done, self._tasks_pending = await asyncio.wait(
                    self._tasks_running,
                    return_when="FIRST_COMPLETED",
                )

    def _create_tasks(self, args_list: list):
        assert isinstance(self.loop, asyncio.AbstractEventLoop)

        for args in args_list:
            if (
                self.maximum_num_jobs_submitted > 0
                and self.num_jobs_submitted >= self.maximum_num_jobs_submitted
            ):
                logging.info(
                    f"Maximum number of jobs to spawn reached ({self.maximum_num_jobs_submitted})"
                )
                raise MaximumJobsSpawnReached

            # Create a Job object from the input arguments
            job_id = self._storage.create_new_job(self._search_id)
            self._storage.store_job_in(job_id, args=(args,))
            new_job = self._create_job(job_id, args, self.run_function, self._storage)
            self.job_id_submitted.append(job_id)

            # Set the context of the job
            # TODO: the notion of `search` in the storage should probably be updated to something
            # TODO: like `group` or `campaign` because can be used in different context than search
            new_job.context.search = self.search

            self.jobs.append(new_job)

            self._on_launch(new_job)

            # The task is created and automatically registered in the event loop when
            task = self.loop.create_task(self._execute(new_job))
            self._tasks_running.append(task)

    def _on_launch(self, job):
        """Called after a job is started."""
        job.status = JobStatus.READY

        job.metadata["timestamp_submit"] = time.time() - self.timestamp

        # Call callbacks
        for cb in self._callbacks:
            cb.on_launch(job)

    def _on_done(self, job):
        """Called after a job has completed."""
        if job.status is JobStatus.RUNNING:
            job.status = JobStatus.DONE

        job.metadata["timestamp_gather"] = time.time() - self.timestamp

        if isinstance(job, HPOJob):
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

        if isinstance(job, HPOJob) and not (isinstance(job.output, dict)):
            raise ValueError(
                "The output of the job is not standard. Check if `job.set_output(output) was "
                "correctly used when defining the Evaluator class."
            )

        return job

    @abc.abstractmethod
    async def execute(self, job: Job) -> Job:
        """Execute the received job. To be implemented with a specific backend.

        Args:
            job (Job): the ``Job`` to be executed.

        Returns:
            job: the update ``Job``.
        """

    def set_event_loop(self):
        if self.loop is None or self.loop.is_closed():
            try:
                # works if `timeout` is not set and code is running in main thread
                self.loop = asyncio.get_running_loop()
            except RuntimeError:
                # required when `timeout` is set because code is not running in main thread
                self.loop = asyncio.new_event_loop()

    def submit(self, args_list: List[Dict]):
        """Send configurations to be evaluated by available workers.

        Args:
            args_list (List[Dict]):
                A list of dict which will be passed to the run function to be executed.
        """
        logging.info(f"submit {len(args_list)} job(s) starts...")
        self.set_event_loop()
        # self.loop.run_until_complete(self._create_tasks(args_list))
        self._create_tasks(args_list)
        logging.info("submit done")

    def gather(self, type, size=1) -> list[Job] | tuple[list[Job], list[Job]]:
        """Collect the completed tasks from the evaluator in batches of one or more.

        Args:
            type (str):
                Options:
                    ``"ALL"``
                        Block until all jobs submitted to the evaluator are completed.
                    ``"BATCH"``
                        Specify a minimum batch size of jobs to collect from
                        the evaluator. The method will block until at least
                        ``size`` evaluations are completed.

            size (int, optional):
                The minimum batch size that we want to collect from the
                evaluator. Defaults to 1.

        Raises:
            Exception: Raised when a gather operation other than "ALL" or "BATCH" is provided.

        Returns:
            list[Job] | tuple[list[Job], list[Job]]: A batch of completed jobs that is at minimum
            the given size.
        """
        logging.info(f"gather({type}, size={size}) starts...")
        assert type in ["ALL", "BATCH"], f"Unsupported gather operation: {type}."
        assert isinstance(self.loop, asyncio.AbstractEventLoop)

        if type == "ALL":
            size = len(self._tasks_running)  # Get all tasks.

        if size > 0:
            self.loop.run_until_complete(self._await_at_least_n_tasks(size))

        local_results = self.process_local_tasks_done(self._tasks_done)

        # Access storage to return results from other processes
        other_results = self.gather_other_jobs_done()

        # call callbacks
        for cb in self._callbacks:
            cb.on_gather(local_results, other_results)

        if len(other_results) == 0:
            logging.info(f"gather done - {len(local_results)} job(s)")

            return local_results
        else:
            logging.info(
                f"gather done - {len(local_results)} local(s) and {len(other_results)} "
                "other(s) job(s)"
            )

            return local_results, other_results

    def gather_other_jobs_done(self):
        """Access storage to return results from other processes."""
        logging.info("gather jobs from other processes")

        job_id_all = self._storage.load_all_job_ids(self._search_id)
        job_id_not_gathered = np.setdiff1d(
            job_id_all,
            self.job_id_submitted + self.job_id_gathered,
        ).tolist()

        other_results = []
        if len(job_id_not_gathered) > 0:
            jobs_data = self._storage.load_jobs(job_id_not_gathered)

            for job_id in job_id_not_gathered:
                job_data = jobs_data[job_id]
                if job_data and job_data["out"]:
                    job = self._create_job(
                        job_id,
                        job_data["in"]["args"][0],
                        run_function=None,
                        storage=self._storage,
                    )
                    if job.status is JobStatus.RUNNING:
                        job.status = JobStatus.DONE
                    job.metadata.update(job_data["metadata"])
                    job.set_output(job_data["out"])
                    self.job_id_gathered.append(job_id)
                    other_results.append(job)

                    for cb in self._callbacks:
                        cb.on_done_other(job)

        return other_results

    def decode(self, key):
        """Decode the key following a JSON format to return a dict."""
        x = json.loads(key)
        if not isinstance(x, dict):
            raise ValueError(f"Expected dict, but got {type(x)}")
        return x

    def process_local_tasks_done(self, tasks):
        local_results = []
        for task in tasks:
            if task.cancelled():
                continue

            job = task.result()
            self._on_done(job)
            local_results.append(job)
            self._tasks_running.remove(task)
            self.job_id_gathered.append(job.id)
            self.job_id_submitted.remove(job.id)

        self._tasks_done = []
        self._tasks_pending = []

        return local_results

    def close(self) -> List[Job]:
        logging.info(f"Closing {type(self).__name__}")
        jobs = []
        if self.loop is None:
            # calling callbacks
            for cb in self._callbacks:
                cb.on_close()

            logging.info(f"{type(self).__name__} closed")
            return jobs

        # Attempt to close tasks in loop
        if not self.loop.is_closed():
            for t in self._tasks_running:
                t.cancel()

            # Wait for tasks to be canceled
            if len(self._tasks_running) > 0:
                self.gather("ALL")

                for job in self.jobs:
                    if job.status in [JobStatus.READY, JobStatus.RUNNING]:
                        job.status = JobStatus.CANCELLED

                        if isinstance(job, HPOJob):
                            job.set_output("F_CANCELLED")

                        self._on_done(job)
                        self.job_id_gathered.append(job.id)
                        jobs.append(job)

                for cb in self._callbacks:
                    cb.on_gather(jobs, [])

        self._tasks_done = []
        self._tasks_pending = []

        # Attempt to close loop if not running
        if not self.loop.is_running():
            self.loop.close()
            self.loop = None

        # calling callbacks
        for cb in self._callbacks:
            cb.on_close()

        logging.info(f"{type(self).__name__} closed")
        return jobs

    def __del__(self):
        if hasattr(self, "loop"):
            self.close()

    def _update_job_when_done(self, job: Job, output) -> Job:
        # Check if the output is a Job object or else
        # If the output is a Job object it means that the run_function is for example
        # following a Producer-Consumer pattern.
        if isinstance(output, Job):
            job = output
        else:
            job.set_output(output)
        return job

    @property
    def is_master(self):
        """Boolean that indicates if the current Evaluator object is a "Master"."""
        return True

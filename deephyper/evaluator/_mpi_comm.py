import asyncio
import functools
import logging
import traceback

from typing import Callable, Hashable

from deephyper.core.exceptions import RunFunctionError
from deephyper.evaluator._evaluator import Evaluator
from deephyper.evaluator._job import Job
from deephyper.evaluator.storage import Storage

import mpi4py

# !To avoid initializing MPI when module is imported (MPI is optional)
mpi4py.rc.initialize = False
mpi4py.rc.finalize = True
from mpi4py import MPI  # noqa: E402
from mpi4py.futures import MPICommExecutor  # noqa: E402

logger = logging.getLogger(__name__)


def catch_exception(run_func):
    """A wrapper function to execute the ``run_func`` passed by the user. This way we can catch remote exception"""
    try:
        code = 0
        result = run_func()
    except Exception:
        code = 1
        result = traceback.format_exc()
    return code, result


class MPICommEvaluator(Evaluator):
    """This evaluator uses the ``mpi4py`` library as backend.

    This evaluator consider an already existing MPI-context (with running processes), therefore it has less overhead than ``MPIPoolEvaluator`` which spawn processes dynamically.

    Args:
        run_function (callable): functions to be executed by the ``Evaluator``.
        num_workers (int, optional): Number of parallel Ray-workers used to compute the ``run_function``. Defaults to ``None`` which consider 1 rank as a worker (minus the master rank).
        callbacks (list, optional): A list of callbacks to trigger custom actions at the creation or completion of jobs. Defaults to ``None``.
        run_function_kwargs (dict, optional): Keyword-arguments to pass to the ``run_function``. Defaults to ``None``.
        comm (optional): A MPI communicator, if ``None`` it will use ``MPI.COMM_WORLD``. Defaults to ``None``.
        rank (int, optional): The rank of the master process. Defaults to ``0``.
        abort_on_exit (bool): If ``True`` then it will call ``comm.Abort()`` to force all MPI processes to finish when closing the ``Evaluator`` (i.e., exiting the current ``with`` block).
    """

    def __init__(
        self,
        run_function: Callable,
        num_workers: int = None,
        callbacks=None,
        run_function_kwargs=None,
        storage: Storage = None,
        search_id: Hashable = None,
        comm=None,
        root=0,
        abort_on_exit=False,
        wait_on_exit=True,
        cancel_jobs_on_exit=True,
    ):
        super().__init__(
            run_function=run_function,
            num_workers=num_workers,
            callbacks=callbacks,
            run_function_kwargs=run_function_kwargs,
            storage=storage,
            search_id=search_id,
        )
        if not MPI.Is_initialized():
            MPI.Init_thread()

        self.comm = comm if comm else MPI.COMM_WORLD
        self.root = root
        self.abort_on_exit = abort_on_exit
        self.wait_on_exit = wait_on_exit
        self.cancel_jobs_on_exit = cancel_jobs_on_exit
        self.num_workers = self.comm.Get_size() - 1  # 1 rank is the master
        self.sem = asyncio.Semaphore(self.num_workers)
        logging.info(f"Creating MPICommExecutor with {self.num_workers} max_workers...")
        self.executor = MPICommExecutor(comm=self.comm, root=self.root)
        self.master_executor = None
        logging.info("Creation of MPICommExecutor done")

    def __enter__(self):
        self.master_executor = self.executor.__enter__()
        if self.master_executor is not None:
            return self
        else:
            return None

    def __exit__(self, type, value, traceback):
        if self.abort_on_exit:
            self.comm.Abort(1)
        else:
            if (
                self.master_executor
                and hasattr(self.executor, "_executor")
                and self.executor._executor is not None
            ):
                self.executor._executor.shutdown(
                    wait=self.wait_on_exit, cancel_futures=self.cancel_jobs_on_exit
                )
                self.executor._executor = None

    async def execute(self, job: Job) -> Job:
        async with self.sem:

            running_job = job.create_running_job(self._storage, self._stopper)

            run_function = functools.partial(
                job.run_function, running_job, **self.run_function_kwargs
            )

            code, sol = await self.loop.run_in_executor(
                self.master_executor, catch_exception, run_function
            )

            # check if exception happened in worker
            if code == 1:
                if "SearchTerminationError" in sol:
                    pass
                else:
                    format_msg = "\n\n/**** START OF REMOTE ERROR ****/\n\n"
                    format_msg += sol
                    format_msg += "\nException happening in remote rank was propagated to root process.\n"
                    format_msg += "\n/**** END OF REMOTE ERROR ****/\n"
                    raise RunFunctionError(format_msg)

            job.set_output(sol)

        return job

import asyncio
import functools
import logging
import traceback
from typing import Callable, Hashable

from deephyper.evaluator import Evaluator, Job, JobStatus
from deephyper.evaluator.mpi import MPI, MPICommExecutor
from deephyper.evaluator.storage import Storage
from deephyper.evaluator.storage._mpi_win_storage import MPIWinStorage

logger = logging.getLogger(__name__)


def catch_exception(run_func):
    """A wrapper function to execute the ``run_func`` passed by the user.

    This is used to catch remote exception.
    """
    try:
        code = 0
        result = run_func()
    except Exception:
        code = 1
        result = traceback.format_exc()
    print(f"{code=}, {result=}")
    return code, result


class MPICommEvaluator(Evaluator):
    """This evaluator uses the ``mpi4py`` library as backend.

    This evaluator consider an already existing MPI-context (with running
    processes), therefore it has less overhead than ``MPIPoolEvaluator``
    which spawn processes dynamically.

    Args:
        run_function (callable):
            Functions to be executed by the ``Evaluator``.
        num_workers (int, optional):
            Number of parallel Ray-workers used to compute the
            ``run_function``. Defaults to ``None`` which consider 1 rank as a
            worker (minus the master rank).
        callbacks (list, optional):
            A list of callbacks to trigger custom actions at the creation or
            completion of jobs. Defaults to ``None``.
        run_function_kwargs (dict, optional):
            Keyword-arguments to pass to the ``run_function``. Defaults to ``None``.
        storage (Storage, optional):
            Storage used by the evaluator. Defaults to ``SharedMemoryStorage``.
        search_id (Hashable, optional):
            The id of the search to use in the corresponding storage. If
            ``None`` it will create a new search identifier when initializing
            the search.
        comm (optional):
            A MPI communicator, if ``None`` it will use ``MPI.COMM_WORLD``. Defaults to ``None``.
        rank (int, optional):
            The rank of the master process. Defaults to ``0``.
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
    ):
        if not MPI.Is_initialized():
            MPI.Init_thread()

        self.comm = comm if comm else MPI.COMM_WORLD
        self.root = root

        if storage is None:
            logging.info(
                f"No storage was given to create {type(self).__name__} so using MPIWinStorage"
            )
            storage = MPIWinStorage(self.comm, root=self.root)

        if isinstance(storage, MPIWinStorage):
            if search_id is None:
                logging.info(
                    "No search_id was given and an MPIWinStorage is used. Creating new search."
                )
                if self.comm.Get_rank() == self.root:
                    search_id = storage.create_new_search()
        self.comm.Barrier()

        super().__init__(
            run_function=run_function,
            num_workers=num_workers,
            callbacks=callbacks,
            run_function_kwargs=run_function_kwargs,
            storage=storage,
            search_id=search_id,
        )

        self.num_workers = self.comm.Get_size() - 1  # 1 rank is the master
        self.sem = None
        logging.info(f"Creating MPICommExecutor with {self.num_workers} max_workers...")

        if self.num_workers == 0 and self.comm.Get_size() <= 1:
            raise RuntimeError(
                "No workers was detected because there was only 1 MPI rank. The number of MPI "
                "ranks must be greater than 1."
            )

        self._comm_executor = None
        self._pool_executor = None
        logging.info("Creation of MPICommExecutor done")

    def set_event_loop(self):
        super().set_event_loop()
        # The semaphore should be created after getting the event loop to avoid
        # binding it to a different event loop
        self.sem = asyncio.Semaphore(self.num_workers)

    @property
    def is_master(self):
        return self.comm.Get_rank() == self.root

    def __enter__(self):
        self._comm_executor = MPICommExecutor(comm=self.comm, root=self.root)
        self._pool_executor = self._comm_executor.__enter__()
        return self

    def __exit__(self, type, value, traceback):
        if self.is_master:
            if self.loop is not None and not self.loop.is_closed():
                self.close()
            self._pool_executor.__exit__(type, value, traceback)
            self._pool_executor = None

    async def execute(self, job: Job) -> Job:
        async with self.sem:
            job.status = JobStatus.RUNNING

            running_job = job.create_running_job(self._stopper)

            run_function = functools.partial(
                job.run_function, running_job, **self.run_function_kwargs
            )

            run_function_future = self.loop.run_in_executor(self._pool_executor, run_function)

            if self.timeout is not None:
                try:
                    output = await asyncio.wait_for(
                        asyncio.shield(run_function_future), timeout=self.time_left
                    )
                except asyncio.TimeoutError:
                    job.status = JobStatus.CANCELLING
                    output = await run_function_future
                    job.status = JobStatus.CANCELLED
            else:
                output = await run_function_future

            return self._update_job_when_done(job, output)

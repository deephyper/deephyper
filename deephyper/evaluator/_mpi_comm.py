import asyncio
import functools
import logging
from deephyper.evaluator._evaluator import Evaluator

import mpi4py
mpi4py.rc.initialize = False
mpi4py.rc.finalize = True
from mpi4py import MPI
from mpi4py.futures import MPICommExecutor


logger = logging.getLogger(__name__)


class MPICommEvaluator(Evaluator):
    """This evaluator uses the ``ray`` library as backend.

    Args:
        run_function (callable): functions to be executed by the ``Evaluator``.
        num_workers (int, optional): Number of parallel Ray-workers used to compute the ``run_function``. Defaults to 1.
        callbacks (list, optional): A list of callbacks to trigger custom actions at the creation or completion of jobs. Defaults to None.
    """

    def __init__(
        self,
        run_function,
        num_workers: int = None,
        callbacks=None,
        run_function_kwargs=None,
    ):
        super().__init__(run_function, num_workers, callbacks, run_function_kwargs)
        if not MPI.Is_initialized():
            MPI.Init_thread()

        self.comm = MPI.COMM_WORLD
        self.num_workers = self.comm.Get_size() - 1 # 1 rank is the master
        self.sem = asyncio.Semaphore(self.num_workers)
        logging.info(f"Creating MPICommExecutor with {self.num_workers} max_workers...")
        self.executor = MPICommExecutor(comm=self.comm, root=0)
        self.master_executor = None
        logging.info("Creation of MPICommExecutor done")

    def __enter__(self):
        self.master_executor = self.executor.__enter__()
        if self.master_executor is not None:
            return self
        else:
            return None
    
    def __exit__(self, type, value, traceback):
        self.executor.__exit__(type, value, traceback)
        self.master_executor = None

    async def execute(self, job):
        async with self.sem:

            run_function = functools.partial(
                job.run_function, job.config, **self.run_function_kwargs
            )

            sol = await self.loop.run_in_executor(self.master_executor, run_function)

            job.result = sol

        return job




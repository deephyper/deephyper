import asyncio
import functools
import logging
from deephyper.evaluator._evaluator import Evaluator

from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor


logger = logging.getLogger(__name__)


class MPIPoolEvaluator(Evaluator):
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
        self.comm = MPI.COMM_WORLD
        self.num_workers = self.comm.Get_size() - 1 # 1 rank is the master
        self.sem = asyncio.Semaphore(self.num_workers)
        logging.info(f"Creating MPIPoolExecutor with {self.num_workers} max_workers...")
        self.executor = MPIPoolExecutor(max_workers=self.num_workers)
        logging.info("Creation of MPIPoolExecutor done")

    async def execute(self, job):
        async with self.sem:

            run_function = functools.partial(
                job.run_function, job.config, **self.run_function_kwargs
            )

            sol = await self.loop.run_in_executor(self.executor, run_function)

            job.result = sol

        return job




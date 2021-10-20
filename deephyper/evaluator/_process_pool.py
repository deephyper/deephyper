import logging
import asyncio

from deephyper.evaluator._evaluator import Evaluator

from concurrent.futures import ProcessPoolExecutor

logger = logging.getLogger(__name__)


class ProcessPoolEvaluator(Evaluator):
    """This evaluator uses the ``ProcessPoolExecutor`` as backend.

    Args:
        run_function (callable): functions to be executed by the ``Evaluator``.
        num_workers (int, optional): Number of parallel processes used to compute the ``run_function``. Defaults to 1.
        callbacks (list, optional): A list of callbacks to trigger custom actions at the creation or completion of jobs. Defaults to None.
    """

    def __init__(self, run_function, num_workers: int=1, callbacks=None):
        super().__init__(run_function, num_workers, callbacks)
        self.sem = asyncio.Semaphore(num_workers)
        logger.info(
            f"ProcessPool Evaluator will execute {self.run_function.__name__}() from module {self.run_function.__module__}"
        )

    async def execute(self, job):

        async with self.sem:

            executor = ProcessPoolExecutor(max_workers=1)
            sol = await self.loop.run_in_executor(executor, job.run_function, job.config)

            job.result = sol


        return job
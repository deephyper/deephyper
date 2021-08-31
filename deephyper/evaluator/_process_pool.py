import logging
import time

from deephyper.evaluator._thread_pool import ThreadPoolEvaluator

from concurrent.futures import ProcessPoolExecutor

logger = logging.getLogger(__name__)


class ProcessPoolEvaluator(ThreadPoolEvaluator):

    def __init__(self, run_function, num_workers: int=1, callbacks=None):
        super().__init__(run_function, num_workers, callbacks)
        self.executor = ProcessPoolExecutor(max_workers = num_workers)
        self.n_jobs = 0
        logger.info(
            f"ProcessPool Evaluator will execute {self.run_function.__name__}() from module {self.run_function.__module__}"
        )

    async def execute(self, job):
        sol = await self.loop.run_in_executor(self.executor, job.run_function, job.config)

        job.result = sol

        return job
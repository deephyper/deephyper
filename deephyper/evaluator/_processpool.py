import logging
import asyncio
import time

from deephyper.evaluator._threadpool import ThreadPoolEvaluator

from concurrent.futures import ProcessPoolExecutor

logger = logging.getLogger(__name__)


class ProcessPoolEvaluator(ThreadPoolEvaluator):

    def __init__(self, run_function, method, num_workers=1):
        super().__init__(run_function, method, num_workers)
        self.executor = ProcessPoolExecutor(max_workers = num_workers)
        self.n_jobs = 0
        logger.info(
            f"ProcessPool Evaluator will execute {self.run_function.__name__}() from module {self.run_function.__module__}"
        )    

    async def execute(self, job):

        start_time = time.time()
    
        job.status = job.RUNNING
        sol = await self.loop.run_in_executor(self.executor, job.run_function, job.config)
        job.duration = time.time() - start_time

        job.status = job.DONE

        job.result = (job.config, sol)

        return job
import logging
import time

from deephyper.evaluator.async_evaluate import AsyncEvaluator

logger = logging.getLogger(__name__)


class ThreadPoolEvaluator(AsyncEvaluator):

    def __init__(self, run_function, method, num_workers=1):
        super().__init__(run_function, method, num_workers)
        logger.info(
            f"ThreadPool Evaluator will execute {self.run_function.__name__}() from module {self.run_function.__module__}"
        )
    
    async def execute(self, job):

        start_time = time.time()

        job.status = job.RUNNING
        # The default executor used in the "run_in_executor(...)" if None is given is
        # a ThreadPoolEvaluator.
        sol = await self.loop.run_in_executor(None, job.run_function, job.config)
        job.duration = time.time() - start_time

        job.status = job.DONE

        job.result = (job.config, sol)

        return job

    


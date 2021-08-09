import logging
import time

from deephyper.evaluator.evaluate import Evaluator

logger = logging.getLogger(__name__)


class ThreadPoolEvaluator(Evaluator):

    def __init__(self, run_function, num_workers=1, callbacks=None):
        super().__init__(run_function, num_workers, callbacks)
        logger.info(
            f"ThreadPool Evaluator will execute {self.run_function.__name__}() from module {self.run_function.__module__}"
        )

    async def execute(self, job):

        # The default executor used in the "run_in_executor(...)" if None is given is
        # a ThreadPoolEvaluator.
        sol = await self.loop.run_in_executor(None, job.run_function, job.config)

        job.result = sol

        return job




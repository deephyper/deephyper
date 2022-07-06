import asyncio
import functools
import logging
from concurrent.futures import ThreadPoolExecutor

from deephyper.evaluator._evaluator import Evaluator

logger = logging.getLogger(__name__)


class ThreadPoolEvaluator(Evaluator):
    """This evaluator uses the ``ThreadPoolExecutor`` as backend.

    .. warning:: This evaluator is interesting with I/O intensive tasks, do not expect a speed-up with compute intensive tasks.

    Args:
        run_function (callable): functions to be executed by the ``Evaluator``.
        num_workers (int, optional): Number of concurrent threads used to compute the ``run_function``. Defaults to 1.
        callbacks (list, optional): A list of callbacks to trigger custom actions at the creation or completion of jobs. Defaults to None.
    """

    def __init__(
        self,
        run_function,
        num_workers: int = 1,
        callbacks: list = None,
        run_function_kwargs: dict = None,
    ):
        super().__init__(run_function, num_workers, callbacks, run_function_kwargs)
        self.sem = asyncio.Semaphore(num_workers)
        self.executor = ThreadPoolExecutor(max_workers=num_workers)

        if hasattr(run_function, "__name__") and hasattr(run_function, "__module__"):
            logger.info(
                f"ThreadPool Evaluator will execute {self.run_function.__name__}() from module {self.run_function.__module__}"
            )
        else:
            logger.info(f"Thread Evaluator will execute {self.run_function}")

    async def execute(self, job):
        async with self.sem:

            run_function = functools.partial(
                job.run_function, job.config, **self.run_function_kwargs
            )

            sol = await self.loop.run_in_executor(self.executor, run_function)

            job.result = sol

        return job

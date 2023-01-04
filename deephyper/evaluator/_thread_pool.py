import asyncio
import functools
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Hashable

from deephyper.evaluator._evaluator import Evaluator
from deephyper.evaluator._job import Job
from deephyper.evaluator.storage import Storage

logger = logging.getLogger(__name__)


class ThreadPoolEvaluator(Evaluator):
    """This evaluator uses the ``ThreadPoolExecutor`` as backend.

    .. warning:: This evaluator is interesting with I/O intensive tasks, do not expect a speed-up with compute intensive tasks.

    Args:
        run_function (callable): functions to be executed by the ``Evaluator``.
        num_workers (int, optional): Number of concurrent threads used to compute the ``run_function``. Defaults to 1.
        callbacks (list, optional): A list of callbacks to trigger custom actions at the creation or completion of jobs. Defaults to None.
        run_function_kwargs (dict, optional): Static keyword arguments to pass to the ``run_function`` when executed.
        storage (Storage, optional): Storage used by the evaluator. Defaults to ``MemoryStorage``.
        search_id (Hashable, optional): The id of the search to use in the corresponding storage. If ``None`` it will create a new search identifier when initializing the search.
    """

    def __init__(
        self,
        run_function: Callable,
        num_workers: int = 1,
        callbacks: list = None,
        run_function_kwargs: dict = None,
        storage: Storage = None,
        search_id: Hashable = None,
    ):
        super().__init__(
            run_function=run_function,
            num_workers=num_workers,
            callbacks=callbacks,
            run_function_kwargs=run_function_kwargs,
            storage=storage,
            search_id=search_id,
        )
        self.sem = asyncio.Semaphore(num_workers)
        self.executor = ThreadPoolExecutor(max_workers=num_workers)

        if hasattr(run_function, "__name__") and hasattr(run_function, "__module__"):
            logger.info(
                f"ThreadPool Evaluator will execute {self.run_function.__name__}() from module {self.run_function.__module__}"
            )
        else:
            logger.info(f"Thread Evaluator will execute {self.run_function}")

    async def execute(self, job: Job) -> Job:
        async with self.sem:

            running_job = job.create_running_job(self._storage, self._stopper)

            run_function = functools.partial(
                job.run_function, running_job, **self.run_function_kwargs
            )

            output = await self.loop.run_in_executor(self.executor, run_function)

            job.set_output(output)

        return job

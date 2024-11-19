import asyncio
import functools
import logging
import os
import signal
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Hashable

from deephyper.evaluator import Evaluator, Job, JobStatus
from deephyper.evaluator.storage import SharedMemoryStorage, Storage

logger = logging.getLogger(__name__)


class ProcessPoolEvaluator(Evaluator):
    """This evaluator uses the ``ProcessPoolExecutor`` as backend.

    Args:
        run_function (callable): functions to be executed by the ``Evaluator``.
        num_workers (int, optional): Number of parallel processes used to compute the ``run_function``. Defaults to 1.
        callbacks (list, optional): A list of callbacks to trigger custom actions at the creation or completion of jobs. Defaults to None.
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

        if storage is None:
            storage = SharedMemoryStorage()

        super().__init__(
            run_function=run_function,
            num_workers=num_workers,
            callbacks=callbacks,
            run_function_kwargs=run_function_kwargs,
            storage=storage,
            search_id=search_id,
        )
        self.sem = asyncio.Semaphore(num_workers)

        # !creating the exector once here is crutial to avoid repetitive overheads
        self.executor = ProcessPoolExecutor(
            max_workers=self.num_workers,
        )

        if hasattr(run_function, "__name__") and hasattr(run_function, "__module__"):
            logger.info(
                f"ProcessPool Evaluator will execute {self.run_function.__name__}() from module {self.run_function.__module__}"
            )
        else:
            logger.info(f"ProcessPool Evaluator will execute {self.run_function}")

    async def execute(self, job: Job) -> Job:
        async with self.sem:

            job.status = JobStatus.RUNNING

            running_job = job.create_running_job(self._stopper)

            run_function = functools.partial(
                job.run_function, running_job, **self.run_function_kwargs
            )

            run_function_future = self.loop.run_in_executor(self.executor, run_function)

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

            job.set_output(output)

        return job

    def close(self):
        logging.info("Closing ProcessPoolEvaluator")
        if self.executor._processes is not None:
            for pid in self.executor._processes:
                logging.info(f"Stopping worker process {pid} of ProcessPoolEvaluator")
                os.kill(pid, signal.SIGKILL)
        self.executor.shutdown(wait=True, cancel_futures=True)
        self.executor = ProcessPoolExecutor(
            max_workers=self.num_workers,
        )
        if not self.loop.is_running():
            self.loop.close()

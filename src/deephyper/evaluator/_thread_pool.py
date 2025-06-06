import asyncio
import functools
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Hashable

from deephyper.evaluator import Evaluator, Job, JobStatus
from deephyper.evaluator.storage import Storage

logger = logging.getLogger(__name__)


class ThreadPoolEvaluator(Evaluator):
    """This evaluator uses the ``ThreadPoolExecutor`` as backend.

    .. warning::

       This evaluator is interesting with I/O intensive tasks, do not expect a
       speed-up with compute intensive tasks.

    Args:
        run_function (callable):
            Functions to be executed by the ``Evaluator``.
        num_workers (int, optional):
            Number of concurrent threads used to compute the ``run_function``.
            Defaults to 1.
        callbacks (list, optional):
            A list of callbacks to trigger custom actions at the creation or
            completion of jobs. Defaults to None.
        run_function_kwargs (dict, optional):
            Static keyword arguments to pass to the ``run_function`` when executed.
        storage (Storage, optional):
            Storage used by the evaluator. Defaults to ``MemoryStorage``.
        search_id (Hashable, optional):
            The id of the search to use in the corresponding storage. If
            ``None`` it will create a new search identifier when initializing
            the search.
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
        self.executor = ThreadPoolExecutor(
            max_workers=num_workers,
            thread_name_prefix="ThreadPoolEvaluator",
        )
        self.sem = None

    def set_event_loop(self):
        super().set_event_loop()
        # The semaphore should be created after getting the event loop to avoid
        # binding it to a different event loop
        self.sem = asyncio.Semaphore(self.num_workers)

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

            return self._update_job_when_done(job, output)

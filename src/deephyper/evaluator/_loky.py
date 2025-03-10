import asyncio
import functools
from typing import Callable, Hashable

import cloudpickle

from loky import ProcessPoolExecutor

from deephyper.evaluator import Evaluator, Job, JobStatus
from deephyper.evaluator.storage import SharedMemoryStorage, Storage


def run_with_cloudpickle(loop, executor, func, *args):
    # Serialize the function and arguments using cloudpickle
    serialized_task = cloudpickle.dumps((func, args))

    def wrapped_task(serialized_task):
        func, args = cloudpickle.loads(serialized_task)
        return func(*args)

    return loop.run_in_executor(executor, wrapped_task, serialized_task)


class LokyEvaluator(Evaluator):
    """This evaluator uses the ``ProcessPoolExecutor`` from ``loky`` as backend.

    The ``loky`` backend uses ``cloudpickle`` to serialize by value the called function and
    its arguments.

    Args:
        run_function (callable):
            Functions to be executed by the ``Evaluator``.
        num_workers (int, optional):
            Number of parallel processes used to compute the ``run_function``. Defaults to 1.
        callbacks (list, optional):
            A list of callbacks to trigger custom actions at the creation or
            completion of jobs. Defaults to None.
        run_function_kwargs (dict, optional):
            Static keyword arguments to pass to the ``run_function`` when executed.
        storage (Storage, optional):
            Storage used by the evaluator. Defaults to ``SharedMemoryStorage``.
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

        # Creating the exector once here is crutial to avoid repetitive overheads
        self.executor = ProcessPoolExecutor(
            max_workers=self.num_workers,
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

            # run_function_future = self.loop.run_in_executor(self.executor, run_function)
            run_function_future = run_with_cloudpickle(self.loop, self.executor, run_function)

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

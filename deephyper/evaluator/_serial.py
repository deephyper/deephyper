import logging

from typing import Callable

from deephyper.evaluator._evaluator import Evaluator
from deephyper.evaluator._job import Job
from deephyper.evaluator.storage import Storage

logger = logging.getLogger(__name__)


class SerialEvaluator(Evaluator):
    """This evaluator run evaluations one after the other (not parallel).

    Args:
        run_function (callable): functions to be executed by the ``Evaluator``.
        num_workers (int, optional): Number of parallel Ray-workers used to compute the ``run_function``. Defaults to 1.
        callbacks (list, optional): A list of callbacks to trigger custom actions at the creation or completion of jobs. Defaults to None.
    """

    def __init__(
        self,
        run_function: Callable,
        num_workers: int = 1,
        callbacks: list = None,
        run_function_kwargs: dict = None,
        storage: Storage = None,
    ):
        super().__init__(
            run_function, num_workers, callbacks, run_function_kwargs, storage
        )

        self.num_workers = num_workers

        if hasattr(run_function, "__name__") and hasattr(run_function, "__module__"):
            logger.info(
                f"Serial Evaluator will execute {self.run_function.__name__}() from module {self.run_function.__module__}"
            )
        else:
            logger.info(f"Serial Evaluator will execute {self.run_function}")

    async def execute(self, job: Job) -> Job:

        running_job = job.create_running_job(self._storage, self._stopper)

        output = self.run_function(running_job, **self.run_function_kwargs)

        job.set_output(output)

        return job

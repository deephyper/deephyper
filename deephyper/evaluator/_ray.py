import logging
import ray

from typing import Callable, Hashable
from deephyper.evaluator._evaluator import Evaluator
from deephyper.evaluator._job import Job
from deephyper.evaluator.storage import Storage

ray_initializer = None

logger = logging.getLogger(__name__)


class RayEvaluator(Evaluator):
    """This evaluator uses the ``ray`` library as backend.

    Args:
        run_function (callable): functions to be executed by the ``Evaluator``.
        callbacks (list, optional): A list of callbacks to trigger custom actions at the creation or completion of jobs. Defaults to None.
        run_function_kwargs (dict, optional): Static keyword arguments to pass to the ``run_function`` when executed.
        storage (Storage, optional): Storage used by the evaluator. Defaults to ``MemoryStorage``.
        search_id (Hashable, optional): The id of the search to use in the corresponding storage. If ``None`` it will create a new search identifier when initializing the search.
        address (str, optional): address of the Ray-head. Defaults to None, if no Ray-head was started.
        password (str, optional): password to connect ot the Ray-head. Defaults to None, if the default Ray-password is used.
        num_cpus (int, optional): number of CPUs available in the Ray-cluster. Defaults to None, if the Ray-cluster was already started it will be automatically computed.
        num_gpus (int, optional): number of GPUs available in the Ray-cluster. Defaults to None, if the Ray-cluster was already started it will be automatically computed.
        num_cpus_per_task (float, optional): number of CPUs used per remote task. Defaults to 1.
        num_gpus_per_task (float, optional): number of GPUs used per remote task. Defaults to None.
        ray_kwargs (dict, optional): other ray keyword arguments passed to ``ray.init(...)``. Defaults to {}.
        num_workers (int, optional): number of workers available to compute remote-tasks in parallel. Defaults to ``None``, or if it is ``-1`` it is automatically computed based with ``num_workers = int(num_cpus // num_cpus_per_task)``.
    """

    def __init__(
        self,
        run_function: Callable,
        callbacks: list = None,
        run_function_kwargs: dict = None,
        storage: Storage = None,
        search_id: Hashable = None,
        address: str = None,
        password: str = None,
        num_cpus: int = None,
        num_gpus: int = None,
        include_dashboard: bool = False,
        num_cpus_per_task: float = 1,
        num_gpus_per_task: float = None,
        ray_kwargs: dict = None,
        num_workers: int = None,
    ):
        super().__init__(
            run_function=run_function,
            num_workers=num_workers,
            callbacks=callbacks,
            run_function_kwargs=run_function_kwargs,
            storage=storage,
            search_id=search_id,
        )
        # get the __init__ parameters
        self._init_params = locals()

        ray_kwargs = {} if ray_kwargs is None else ray_kwargs
        if address is not None:
            ray_kwargs["address"] = address
        if password is not None:
            ray_kwargs["_redis_password"] = password
        if num_cpus is not None:
            ray_kwargs["num_cpus"] = num_cpus
        if num_gpus is not None:
            ray_kwargs["num_gpus"] = num_gpus
        if include_dashboard is not None:
            ray_kwargs["include_dashboard"] = include_dashboard

        if not (ray.is_initialized()):
            ray.init(**ray_kwargs)

        self.num_cpus_per_task = num_cpus_per_task
        self.num_gpus_per_task = num_gpus_per_task

        self.num_cpus = int(
            sum([node["Resources"].get("CPU", 0) for node in ray.nodes()])
        )
        self.num_gpus = int(
            sum([node["Resources"].get("GPU", 0) for node in ray.nodes()])
        )
        if self.num_workers is None or self.num_workers == -1:
            self.num_workers = int(self.num_cpus // self.num_cpus_per_task)

        if hasattr(run_function, "__name__") and hasattr(run_function, "__module__"):
            logger.info(
                f"Ray Evaluator will execute {self.run_function.__name__}() from module {self.run_function.__module__}"
            )
        else:
            logger.info(f"Ray Evaluator will execute {self.run_function}")

        self._remote_run_function = ray.remote(
            num_cpus=self.num_cpus_per_task,
            num_gpus=self.num_gpus_per_task,
            # max_calls=1,
        )(self.run_function)

    async def execute(self, job: Job) -> Job:

        running_job = job.create_running_job(self._storage, self._stopper)

        output = await self._remote_run_function.remote(
            running_job, **self.run_function_kwargs
        )

        job.set_output(output)

        return job

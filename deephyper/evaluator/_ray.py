import logging
import ray
import time
from deephyper.evaluator.evaluate import Evaluator

ray_initializer = None

logger = logging.getLogger(__name__)

class RayEvaluator(Evaluator):

    def __init__(
        self,
        run_function,
        callbacks=None,
        address: str=None,
        password: str=None,
        num_cpus: int=None,
        num_gpus: int=None,
        num_cpus_per_task: float=1,
        num_gpus_per_task: float=None,
        ray_kwargs: dict={},
        num_workers: int=None,
        ):
        super().__init__(run_function, num_workers, callbacks)

        if address is not None:
            ray_kwargs["address"] = address
        if password is not None:
            ray_kwargs["_redis_password"] = password
        if num_cpus is not None:
            ray_kwargs["num_cpus"] = num_cpus
        if num_gpus is not None:
            ray_kwargs["num_gpus"] = num_gpus

        if not(ray.is_initialized()):
            ray.init(**ray_kwargs)

        self.num_cpus_per_task = num_cpus_per_task
        self.num_gpus_per_task = num_gpus_per_task

        self.num_cpus = int(
            sum([node["Resources"].get("CPU", 0) for node in ray.nodes()])
        )
        self.num_gpus = int(
            sum([node["Resources"].get("GPU", 0) for node in ray.nodes()])
        )
        if self.num_workers is None:
            self.num_workers = int(self.num_cpus // self.num_cpus_per_task)

        logger.info(
            f"Ray Evaluator will execute {self.run_function.__name__}() from module {self.run_function.__module__}"
        )

        self._remote_run_function = ray.remote(
            num_cpus=self.num_cpus_per_task,
            num_gpus=self.num_gpus_per_task,
            # max_calls=1,
        )(self.run_function)

    async def execute(self, job):

        sol = await self._remote_run_function.remote(job.config)

        job.result = sol

        return job


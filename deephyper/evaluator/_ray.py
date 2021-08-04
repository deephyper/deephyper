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
        ray_address=None,
        ray_password=None,
        num_cpus=None,
        num_gpus=None,
        num_cpus_per_task=1,
        num_gpus_per_task=None,
        num_workers=None,
        ):
        super().__init__(run_function, num_workers)

        ray_kwargs = {}
        if ray_address is not None:
            ray_kwargs["address"] = ray_address
        if ray_password is not None:
            ray_kwargs["_redis_password"] = ray_password
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
        self.num_workers = self.num_cpus // self.num_cpus_per_task



        logger.info(
            f"Ray Evaluator will execute {self.run_function.__name__}() from module {self.run_function.__module__}"
        )

    async def execute(self, job):

        sol = await ray.remote(
            num_cpus=job.num_cpus,
            num_gpus=job.num_gpus,
            max_calls=1,
        )(job.run_function).remote(job.config)

        job.result = sol

        return job


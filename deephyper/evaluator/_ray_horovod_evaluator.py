import logging
import traceback
import time
from collections import defaultdict, namedtuple

import ray
from horovod.ray import RayExecutor

from deephyper.evaluator.evaluate import Evaluator
from deephyper.evaluator._ray_evaluator import RayFuture

logger = logging.getLogger(__name__)


class RayHorovodFuture(RayFuture):
    FAIL_RETURN_VALUE = Evaluator.FAIL_RETURN_VALUE

    def __init__(self, func, x, num_slots):
        self.num_slots = num_slots
        self.executor = self.start_executor()
        self.id_res = self.executor.run_remote(func, args=[x])[0]
        self._state = "active"
        self._result = None

    def start_executor(self):
        # Ray executor settings
        setting = RayExecutor.create_settings(timeout_s=100)
        num_hosts = 1  # number of machine to use
        num_slots = self.num_slots  # number of workers to use on each machine
        cpus_per_slot = 1  # number of cores to allocate to each worker
        gpus_per_slot = 1  # number of GPUs to allocate to each worker
        use_gpu = gpus_per_slot > 0

        # Start num_hosts * num_slots actors on the cluster
        # https://horovod.readthedocs.io/en/stable/api.html#horovod-ray-api
        executor = RayExecutor(
            setting,
            num_hosts=num_hosts,
            num_slots=num_slots,
            cpus_per_slot=cpus_per_slot,
            gpus_per_slot=gpus_per_slot,
            use_gpu=use_gpu
        )

        # Launch the Ray actors on each machine
        # This will launch `num_slots` actors on each machine
        executor.start()
        return executor

    def _poll(self):
        if not self._state == "active":
            return

        id_done, _ = ray.wait([self.id_res], num_returns=1, timeout=0.001)

        if len(id_done) == 1:
            try:
                self._result = ray.get(id_done[0])
                self.executor.shutdown()
                self._state = "done"
            except Exception:
                print(traceback.format_exc())
                self._state = "failed"
        else:
            self._state = "active"


class RayHorovodEvaluator(Evaluator):
    """TODO
    """

    WaitResult = namedtuple("WaitResult", ["active", "done", "failed", "cancelled"])

    def __init__(
        self,
        run_function,
        cache_key=None,
        ray_address=None,
        ray_password=None,
        num_cpus_per_task=1,
        num_gpus_per_task=None,
        **kwargs,
    ):
        super().__init__(run_function, cache_key, **kwargs)

        logger.info(f"RAY Evaluator init: redis-address={ray_address}")

        if not ray_address is None:
            proc_info = ray.init(address=ray_address, _redis_password=ray_password)
        else:
            proc_info = ray.init()

        self.num_cpus_per_task = num_cpus_per_task
        self.num_gpus_per_task = num_gpus_per_task
        self.num_slots = self.num_gpus_per_task

        self.num_cpus = int(
            sum([node["Resources"].get("CPU", 0) for node in ray.nodes()])
        )
        self.num_gpus = int(
            sum([node["Resources"].get("GPU", 0) for node in ray.nodes()])
        )

        # TODO: verify the coherence of num_cpus_per_task and num_gpus_per_task
        self.num_workers = self.num_cpus // self.num_cpus_per_task

        logger.info(
            f"RAY Evaluator will execute: '{self._run_function}', proc_info: {proc_info}"
        )

    def _eval_exec(self, x: dict):
        assert isinstance(x, dict)
        future = RayHorovodFuture(self._run_function, x, self.num_slots)
        return future

    @staticmethod
    def _timer(timeout):
        if timeout is None:
            return lambda: True
        else:
            timeout = max(float(timeout), 0.01)
            start = time.time()
            return lambda: (time.time() - start) < timeout

    def wait(self, futures, timeout=None, return_when="ANY_COMPLETED"):
        assert return_when.strip() in ["ANY_COMPLETED", "ALL_COMPLETED"]
        waitall = bool(return_when.strip() == "ALL_COMPLETED")

        num_futures = len(futures)
        active_futures = [f for f in futures if f.active]
        time_isLeft = self._timer(timeout)

        if waitall:

            def can_exit():
                return len(active_futures) == 0

        else:

            def can_exit():
                return len(active_futures) < num_futures

        while time_isLeft():
            if can_exit():
                break
            else:
                active_futures = [f for f in futures if f.active]
                time.sleep(0.04)

        if not can_exit():
            raise TimeoutError(
                f"{timeout} sec timeout expired while "
                f"waiting on {len(futures)} tasks until {return_when}"
            )

        results = defaultdict(list)
        for f in futures:
            results[f._state].append(f)
        return self.WaitResult(
            active=results["active"],
            done=results["done"],
            failed=results["failed"],
            cancelled=results["cancelled"],
        )

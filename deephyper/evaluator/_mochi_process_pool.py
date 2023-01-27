import logging
import asyncio
import functools
import collections

import pymargo
import pymargo.core

from concurrent.futures import ProcessPoolExecutor
from deephyper.evaluator._evaluator import Evaluator

import mpi4py

# !To avoid initializing MPI when module is imported (MPI is optional)
mpi4py.rc.initialize = False
mpi4py.rc.finalize = True
from mpi4py import MPI  # noqa: E402


logger = logging.getLogger(__name__)


def margo_client(protocol, target_address, func, *args, **kwargs):
    with pymargo.core.Engine(protocol, mode=pymargo.client) as engine:
        execute_function = engine.register("execute_function")
        address = engine.lookup(target_address)
        response = execute_function.on(address)(func, *args, **kwargs)
    return response


def execute_function(handle: pymargo.core.Handle, func, *args, **kwargs):
    res = func(*args, **kwargs)
    handle.respond(res)


def margo_server(comm, protocol):
    with pymargo.core.Engine(protocol, mode=pymargo.server) as engine:
        comm.send(engine.address)  # !temporary
        engine.register("execute_function", execute_function)
        engine.enable_remote_shutdown()
        engine.wait_for_finalize()


class MochiEvaluator(Evaluator):
    """This evaluator uses the ``ProcessPoolExecutor`` as backend.

    Args:
        run_function (callable): functions to be executed by the ``Evaluator``.
        num_workers (int, optional): Number of parallel processes used to compute the ``run_function``. Defaults to 1.
        callbacks (list, optional): A list of callbacks to trigger custom actions at the creation or completion of jobs. Defaults to None.
    """

    def __init__(
        self,
        run_function,
        num_workers: int = 1,
        callbacks: list = None,
        run_function_kwargs: dict = None,
        protocol="tcp",
    ):
        super().__init__(run_function, num_workers, callbacks, run_function_kwargs)

        self._protocol = protocol

        # !use of MPI is temporary to initialise addresses
        if not MPI.Is_initialized():
            MPI.Init_thread()

        self._comm = MPI.COMM_WORLD
        self._rank = self._comm.Get_rank()
        self._size = self._comm.Get_size()
        self.executor = None
        if self._rank == 0:  # master

            self.sem = asyncio.Semaphore(num_workers)
            # !creating the exector once here is crutial to avoid repetitive overheads
            self.executor = ProcessPoolExecutor(max_workers=num_workers)

            if hasattr(run_function, "__name__") and hasattr(
                run_function, "__module__"
            ):
                logger.info(
                    f"Mochi Evaluator will execute {self.run_function.__name__}() from module {self.run_function.__module__}"
                )
            else:
                logger.info(f"Mochi Evaluator will execute {self.run_function}")

            # queue of worker addresses
            self._worker_addresses = []
            for i in range(1, self._size):
                address = self._comm.recv(source=i)
                self._worker_addresses.append(address)
            self._qworker_addresses = collections.deque(self._worker_addresses)

        else:  # workers

            margo_server(self._comm, self._protocol)

    def __enter__(self):
        if self.executor:
            self.executor = self.executor.__enter__()
            return self
        else:
            return None

    def __exit__(self, type, value, traceback):
        if self.executor:
            self.executor.__exit__(type, value, traceback)

            # shutdown pymargo servers
            with pymargo.core.Engine(self._protocol, mode=pymargo.client) as engine:
                for target_address in self._worker_addresses:
                    address = engine.lookup(target_address)
                    address.shutdown()

    async def execute(self, job):

        async with self.sem:

            target_address = self._qworker_addresses.popleft()

            running_job = job.create_running_job(self._storage, self._stopper)

            run_function = functools.partial(
                margo_client,
                self._protocol,
                target_address,
                running_job,
                **self.run_function_kwargs,
            )

            sol = await self.loop.run_in_executor(self.executor, run_function)

            job.result = sol

            self._qworker_addresses.append(target_address)

        return job

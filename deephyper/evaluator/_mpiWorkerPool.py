import logging
import os
from mpi4py import MPI
from collections import namedtuple

from deephyper.evaluator.evaluate import Evaluator

logger = logging.getLogger(__name__)
WaitResult = namedtuple('WaitResult', ['active', 'done', 'failed', 'cancelled'])

class MPIFuture():

    def __init__(self, worker, request):
        self._worker = worker
        self._request = request

    def result(self):
        return self._value

    def set_result(self, value):
        self._value = value

class MPIWorkerPool(Evaluator):
    """Evaluator using a pool of MPI workers.

    Args:
        run_function (func): takes one parameter of type dict and returns a scalar value.
        cache_key (func): takes one parameter of type dict and returns a hashable type,
                          used as the key for caching evaluations. Multiple inputs that
                          map to the same hashable key will only be evaluated once.
                          If ``None``, then cache_key defaults to a lossless (identity)
                          encoding of the input dict.
    """
    def __init__(self, run_function, cache_key=None):
        super().__init__(run_function, cache_key)
        self.comm = MPI.COMM_WORLD
        self.num_workers = self.comm.Get_size()-1
        self.avail_workers = [ x+1 for x in range(0, self.num_workers) ]
        funcName = self._run_function.__name__
        moduleName = self._run_function.__module__
        self.appName = '.'.join((moduleName, funcName))

    def _eval_exec(self, x):
        assert isinstance(x, dict)
        worker = self.avail_workers.pop()
        self.comm.send({'func': self.appName, 'args': x }, dest=worker, tag=0)
        req = self.comm.irecv(source=worker, tag=0)
        future = MPIFuture(worker, req)
        return future

    def wait(self, futures, timeout=None, return_when='ANY_COMPLETED'):
        done, failed, cancelled, active = [],[],[],[]
        if(return_when == 'ALL_COMPLETED'):
            results = MPI.Request.waitall([ f._request for f in futures ])
            for r, f in zip(results, futures):
                f.set_result(r)
            done = futures
        else:
            worker, result = MPI.Request.waitany([ f._request for f in futures ])
            self.avail_workers.append(worker)
            for f in futures:
                if f._worker != worker:
                    active.append(f)
                else:
                    f.set_result(result)
                    done.append(f)
        return WaitResult(
            active=active,
            done=done,
            failed=failed,
            cancelled=cancelled
        )

    def shutdown_workers(self):
        req = []
        for k in range(1, self.comm.Get_size()):
            r = self.comm.isend({'exit': True}, dest=k, tag=0)
            req.append(r)
        MPI.Request.waitall(req)

    def __del__(self):
        self.shutdown_workers()

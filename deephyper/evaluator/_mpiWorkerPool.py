import logging
import os
from mpi4py import MPI
from collections import namedtuple

from deephyper.evaluator.evaluate import Evaluator

logger = logging.getLogger(__name__)
WaitResult = namedtuple('WaitResult', ['active', 'done', 'failed', 'cancelled'])


class MPIFuture():

    def __init__(self, cmd):
        self._worker = None
        self._request = None
        self._tag = None
        self._cmd = cmd

    @property
    def posted(self):
        return self._request is not None

    def post(self, comm, worker, tag):
        if(self.posted):
            raise ValueError("Request already posted")
        comm.send(self._cmd, dest=worker, tag=tag)
        self._worker = worker
        self._tag = tag
        self._request = comm.irecv(source=worker, tag=tag)

    @property
    def request(self):
        return self._request

    @property
    def worker(self):
        return self._worker

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
    def __init__(self, run_function, cache_key=None, comm=None, **kwargs):
        super().__init__(run_function, cache_key)
        if(comm is None):
            self.comm = MPI.COMM_WORLD
        else:
            self.comm = comm
        self.num_workers = self.comm.Get_size()-1
        self.avail_workers = [ x+1 for x in range(0, self.num_workers) ]
        funcName = self._run_function.__name__
        moduleName = self._run_function.__module__
        self.appName = '.'.join((moduleName, funcName))

    def _try_posting(self, unposted):
        now_posted = []
        now_unposted = []
        for f in unposted:
            if(len(self.avail_workers) > 0):
                worker = self.avail_workers.pop()
                f.post(self.comm, worker, 0)
                now_posted.append(f)
            else:
                now_unposted.append(f)
        return now_posted, now_unposted

    def _eval_exec(self, x):
        assert isinstance(x, dict)
        cmd = {'cmd': 'exec', 'args': [x] }
        future = MPIFuture(cmd)
        if(len(self.avail_workers) > 0):
            worker = self.avail_workers.pop()
            future.post(self.comm, worker, 0)
        return future

    def wait(self, futures, timeout=None, return_when='ANY_COMPLETED'):
        done, failed, cancelled, active = [],[],[],[]
        posted = [f for f in futures if f.posted]
        unposted = [f for f in futures if not f.posted]

        if(len(posted) == 0):
            newly_posted, unposted = self._try_posting(unposted)
            posted.extend(newly_posted)

        if(return_when == 'ALL_COMPLETED'):
            while(len(posted) > 0 or len(unposted) > 0):
                results = MPI.Request.waitall([ f.request for f in posted ])
                for r, f in zip(results, posted):
                    f.set_result(r)
                    self.avail_workers.append(f.worker)
                done.extend(posted)
                posted, unposted = self._try_posting(unposted)
        else:
            one_completed = False
            for f in posted:
                completed, result = MPI.Request.test(f.request)
                if completed:
                    one_completed = True
                    f.set_result(result)
                    done.append(f)
                    if(len(unposted) > 0):
                        p = unposted.pop(0)
                        p.post(self.comm, worker=f.worker, tag=f.tag)
                        active.append(p)
                    else:
                        self.avail_workers.append(f.worker)
                else:
                    active.append(f)
            if not one_completed:
                status = MPI.Status()
                requests = [ f.request for f in posted]
                idx, result = MPI.Request.waitany(requests, status=status)
                f = posted[idx]
                f.set_result(result)
                done.append(f)
                if(len(unposted) > 0): 
                    p = unposted.pop(0)
                    p.post(self.comm, worker=f.worker, tag=f.tag)
                    active.append(p)
                else:
                    self.avail_workers.append(f.worker)
            for f in unposted:
                active.append(f)

        return WaitResult(
            active=active,
            done=done,
            failed=failed,
            cancelled=cancelled
        )

    def shutdown_workers(self):
        req = []
        for k in range(1, self.comm.Get_size()):
            r = self.comm.isend({'cmd': 'exit'}, dest=k, tag=0)
            req.append(r)
        MPI.Request.waitall(req)

    def __del__(self):
        self.shutdown_workers()

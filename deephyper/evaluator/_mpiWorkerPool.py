import logging
import os
from mpi4py import MPI
from collections import namedtuple

from deephyper.evaluator.evaluate import Evaluator

logger = logging.getLogger(__name__)
WaitResult = namedtuple("WaitResult", ["active", "done", "failed", "cancelled"])


class MPIFuture:
    """MPIFuture is a class meant to track a pending evaluation.
    It record whether it was posted to a worker, the associated
    MPI request, the tag, and the command that was sent."""

    def __init__(self, cmd):
        self._worker = None
        self._request = None
        self._tag = None
        self._cmd = cmd

    @property
    def posted(self):
        """Returns true if the command was posted."""
        return self._request is not None

    def post(self, comm, worker, tag):
        """Posts the request to a particular worker,
        with a particular tag."""
        if self.posted:
            raise ValueError("Request already posted")
        comm.send(self._cmd, dest=worker, tag=tag)
        self._worker = worker
        self._tag = tag
        self._request = comm.irecv(source=worker, tag=tag)

    @property
    def worker(self):
        """Returns the worker to which the request was posted."""
        return self._worker

    @property
    def tag(self):
        """Returns the tag used for this request."""
        return self._tag

    def result(self):
        """Returns the result of the request. This method will
        fail if set_result wasn't called before to set the result."""
        return self._value

    def _set_result(self, value):
        """Sets the result of the request."""
        self._value = value

    def test(self):
        """Tests if the request has completed."""
        completed, result = MPI.Request.test(self._request)
        if completed:
            self._set_result(result)
        return completed

    @staticmethod
    def waitany(futures):
        """Waits for any of the provided futures to complete
        and sets the result of the one that completed."""
        status = MPI.Status()
        requests = [f._request for f in futures]
        idx, result = MPI.Request.waitany(requests, status=status)
        f = futures[idx]
        f._set_result(result)
        return f

    @staticmethod
    def waitall(futures):
        """Waits for all the provided futures to complete and
        sets their result."""
        results = MPI.Request.waitall([f._request for f in futures])
        for r, f in zip(results, futures):
            f._set_result(r)


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

    def __init__(
        self,
        run_function,
        cache_key=None,
        comm=None,
        num_nodes_master=1,
        num_nodes_per_eval=1,
        num_ranks_per_node=1,
        num_evals_per_node=1,
        num_threads_per_rank=64,
        **kwargs
    ):
        """Constructor."""
        super().__init__(run_function, cache_key)
        if comm is None:
            self.comm = MPI.COMM_WORLD
        else:
            self.comm = comm
        self.num_workers = self.comm.Get_size() - 1
        self.avail_workers = []
        for tag in range(0, num_ranks_per_node):
            for rank in range(0, self.num_workers):
                self.avail_workers.append((rank + 1, tag + 1))
        funcName = self._run_function.__name__
        moduleName = self._run_function.__module__
        self.appName = ".".join((moduleName, funcName))

    def _try_posting(self, unposted):
        """This function takes a list of MPIFuture instances that aren't
        posted and try to post as many as possible to available workers,
        returning a pair of lists, the first one containing posted futures,
        the second one containing futures that remained unposted."""
        now_posted = []
        now_unposted = []
        for f in unposted:
            if len(self.avail_workers) > 0:
                worker, tag = self.avail_workers.pop()
                f.post(self.comm, worker, tag)
                now_posted.append(f)
            else:
                now_unposted.append(f)
        return now_posted, now_unposted

    def _eval_exec(self, x):
        """Remotely executes the "exec" function of the MPIWorker
        with the provided point x as argument. Returns an instance
        of MPIFuture. If possible, this future will have been posted."""
        assert isinstance(x, dict)
        cmd = {"cmd": "exec", "args": [x]}
        future = MPIFuture(cmd)
        if len(self.avail_workers) > 0:
            worker, tag = self.avail_workers.pop()
            future.post(self.comm, worker, tag)
        return future

    def wait(self, futures, timeout=None, return_when="ANY_COMPLETED"):
        """Waits for a set of futures to complete. If return_when == ANY_COMPLETED,
        this function will return as soon as at least one of the futures has completed.
        Otherwise it will wait for all the futures to have completed."""
        # TODO: for now the timeout is not taken into account and
        # the failed and cancelled lists will always be empty.
        done, failed, cancelled, active = [], [], [], []
        posted = [f for f in futures if f.posted]
        unposted = [f for f in futures if not f.posted]

        if len(posted) == 0:
            newly_posted, unposted = self._try_posting(unposted)
            posted.extend(newly_posted)

        if return_when == "ALL_COMPLETED":
            while len(posted) > 0 or len(unposted) > 0:
                MPIFuture.waitall(posted)
                for f in posted:
                    self.avail_workers.append((f.worker, f.tag))
                done.extend(posted)
                posted, unposted = self._try_posting(unposted)
        else:  # return_when ==  'ANY_COMPLETED'
            one_completed = False
            # waitany will wait for only one future to complete
            # so we first loop and test all of them before calling
            # waitany if needed.
            for f in posted:
                completed = f.test()
                if completed:
                    one_completed = True
                    done.append(f)
                    # one request completed, try posting a new request
                    if len(unposted) > 0:
                        p = unposted.pop(0)
                        p.post(self.comm, worker=f.worker, tag=f.tag)
                        active.append(p)
                    else:
                        self.avail_workers.append((f.worker, f.tag))
                else:
                    active.append(f)
            if not one_completed:  # we need to call waitany
                f = MPIFuture.waitany(posted)
                done.append(f)
                if len(unposted) > 0:
                    p = unposted.pop(0)
                    p.post(self.comm, worker=f.worker, tag=f.tag)
                    active.append(p)
                else:
                    self.avail_workers.append((f.worker, f.tag))
            for f in unposted:
                active.append(f)

        return WaitResult(active=active, done=done, failed=failed, cancelled=cancelled)

    def shutdown_workers(self):
        """Shuts down all the MPIWorker instances."""
        req = []
        for k in range(1, self.comm.Get_size()):
            r = self.comm.isend({"cmd": "exit"}, dest=k, tag=0)
            req.append(r)
        MPI.Request.waitall(req)

    def __del__(self):
        """Destructor will shutdown all the workers."""
        self.shutdown_workers()

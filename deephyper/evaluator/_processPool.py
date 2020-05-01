import logging
import os
from collections import namedtuple
from concurrent.futures import CancelledError, ProcessPoolExecutor
from concurrent.futures import wait as _futures_wait

from deephyper.evaluator.evaluate import Evaluator

logger = logging.getLogger(__name__)
WaitResult = namedtuple("WaitResult", ["active", "done", "failed", "cancelled"])


class ProcessPoolEvaluator(Evaluator):
    """Evaluator using ProcessPoolExecutor.

    The ProcessPoolEvaluator use the ``concurrent.futures.ProcessPoolExecutor`` class. The processes doesn't share memory but they are forked from the mother process so imports done before are done repeated. Be carefull if your ``run_function`` is loading an package such as tensorflow it can hang.

    Args:
        run_function (func): takes one parameter of type dict and returns a scalar value.
        cache_key (func): takes one parameter of type dict and returns a hashable type, used as the key for caching evaluations. Multiple inputs that map to the same hashable key will only be evaluated once. If ``None``, then cache_key defaults to a lossless (identity) encoding of the input dict.
    """

    def __init__(self, run_function, cache_key=None, **kwargs):
        super().__init__(run_function, cache_key)
        self.num_workers = 1
        self.executor = ProcessPoolExecutor(max_workers=self.num_workers)
        logger.info(
            f"ProcessPool Evaluator will execute {self._run_function.__name__}() from module {self._run_function.__module__}"
        )

    def _eval_exec(self, x):
        assert isinstance(x, dict)
        future = self.executor.submit(self._run_function, x)
        return future

    def wait(self, futures, timeout=None, return_when="ANY_COMPLETED"):
        return_when = return_when.replace("ANY", "FIRST")
        results = _futures_wait(futures, timeout=timeout, return_when=return_when)
        done, failed, cancelled = [], [], []
        active = list(results.not_done)
        if len(active) > 0 and return_when == "ALL_COMPLETED":
            raise TimeoutError(
                f"{timeout} sec timeout expired while "
                f"waiting on {len(futures)} tasks until {return_when}"
            )
        for res in results.done:
            try:
                res.result(timeout=0)
            except CancelledError:
                cancelled.append(res)
            except Exception as e:
                logger.exception("Eval exception:")
                res.set_result(Evaluator.FAIL_RETURN_VALUE)
                res.set_exception(None)
                failed.append(res)
            else:
                done.append(res)
        return WaitResult(active=active, done=done, failed=failed, cancelled=cancelled)

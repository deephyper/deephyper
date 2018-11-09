from collections import namedtuple
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import wait as _futures_wait
from concurrent.futures import CancelledError
import logging
import os
from deephyper.evaluators import evaluate
logger = logging.getLogger(__name__)
WaitResult = namedtuple('WaitResult', ['active', 'done', 'failed', 'cancelled'])

class LocalLiteEvaluator(evaluate.Evaluator):
    def __init__(self, run_function, cache_key=None):
        super().__init__(run_function, cache_key)
        self.num_workers = self.WORKERS_PER_NODE
        self.executor = ProcessPoolExecutor(
            max_workers = self.num_workers
        )
        logger.info(f"Local-Lite Evaluator will execute {self._run_function.__name__}() from module {self._run_function.__module__}")
    
    def _eval_exec(self, x):
        assert isinstance(x, dict)
        future = self.executor.submit(self._run_function, x)
        return future
    
    def wait(self, futures, timeout=None, return_when='ANY_COMPLETED'):
        return_when=return_when.replace('ANY','FIRST')
        results = _futures_wait(futures, timeout=timeout, return_when=return_when)
        done, failed, cancelled = [],[],[]
        active = list(results.not_done)
        if len(active) > 0 and return_when=='ALL_COMPLETED':
            raise TimeoutError(f'{timeout} sec timeout expired while '
            f'waiting on {len(futures)} tasks until {return_when}')
        for res in results.done:
            try: 
                res.result(timeout=0)
            except CancelledError:
                cancelled.append(res)
            except Exception as e:
                logger.exception("Eval exception:")
                res.set_result(evaluate.Evaluator.FAIL_RETURN_VALUE)
                res.set_exception(None)
                failed.append(res)
            else:
                done.append(res)
        return WaitResult(
            active=active,
            done=done,
            failed=failed,
            cancelled=cancelled
        )

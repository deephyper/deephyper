import subprocess
import sys
import logging
from deephyper.evaluators import evaluate
from importlib import import_module
import os
from pprint import pprint
import json

logger = logging.getLogger(__name__)

class PopenFuture:
    def __init__(self, args):
        self.proc = subprocess.Popen(args, shell=True, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, encoding='utf-8')
        self._state = 'active'
        self._result = None

    def _poll(self):
        if not self._state == 'active': return
        retcode = self.proc.poll()
        if retcode is None:
            self._state = 'active'
        elif retcode == 0:
            self._state = 'done'
        else:
            self._state = 'failed'

    def result(self):
        if self._result is not None:
            return self._result
        self.proc.wait()
        if self.done:
            stdout, _ = self.proc.communicate()
            self._result = self._parse(stdout)
        else:
            self._result = self.FAIL_RETURN_VALUE
        return self._result

    def cancel(self):
        self.proc.kill()
        self._state = 'cancelled'

    @property
    def active(self):
        self._poll()
        return self._state == 'active'

    @property
    def done(self):
        self._poll()
        return self._state == 'done'

    @property
    def failed(self):
        self._poll()
        return self._state == 'failed'

    @property
    def cancelled(self):
        self._poll()
        return self._state == 'cancelled'

class LocalEvaluator(evaluate.Evaluator):
    def __init__(self, run_function, cache_key=None):
        super().__init__(run_function, cache_key)
        self.num_workers = self.WORKERS_PER_NODE
        logger.info(f"Local Evaluator to run: {self._run_function}")
    
    def _args(self, x):
        exe = self._runner_executable
        cmd = ' '.join((exe, self.encode(x)))
        return cmd
    
    def wait(self, futures, timeout=None, return_when='ANY_COMPLETED'):

    def _eval_exec(self, x):
        assert isinstance(x, dict)
        cmd = self._args(x)
        future = PopenFuture(cmd)
        logger.info(f"Running: {x}")
        return future

    def _check_done(self):
        return [(key,future) for (key,future) in self.pending_evals.items() if future.done()]

    def get_finished_evals(self):
        '''iter over any immediately available results'''
        done_list = [(key,future) for (key,future) in self.pending_evals.items()
                     if future.done()]

        logger.info(f"{len(done_list)} new evals have completed")
        for key, future in done_list:
            x = self._decode(key)
            y = future.result()
            logger.info(f"x: {x} y: {y}")
            self.evals[key] = y
            del self.pending_evals[key]
            yield (x, y)

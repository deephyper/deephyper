import concurrent.futures
import logging
from deephyper.search import evaluate
from importlib import import_module
import os

logger = logging.getLogger(__name__)

class LocalEvaluator(evaluate.Evaluator):
    ExecutorCls = concurrent.futures.ProcessPoolExecutor

    def __init__(self, params_list, bench_module_name, num_workers=None,
                 backend='tensorflow'):
        super().__init__()
        self.executor = None
        self.num_workers = num_workers
        self.backend = backend
        self.params_list = params_list
        self.bench_module_name = bench_module_name
        self.bench_module = import_module(bench_module_name)
        logger.info("Local Evaluator instantiated")
        logger.info(f"Backend: {self.backend}")
        logger.info(f"Benchmark: {bench_module_name}")

    def _setup_executor(self):
        os.environ['KERAS_BACKEND'] = self.backend
        self.executor = self.ExecutorCls(max_workers=self.num_workers)
        self.num_workers = self.executor._max_workers
        logger.info(f"Created ProcessPool with {self.num_workers} procs")

    def _eval_exec(self, x):
        if self.executor is None:
            self._setup_executor()

        kwargs = {k:v for k,v in zip(self.params_list, x) if 'hidden' not in k}
        eval_func = self.bench_module.run
        future = self.executor.submit(eval_func, **kwargs)
        logger.info(f"Running: {kwargs}")
        return future

    def await_evals(self, to_read):
        '''wait for a set of points to finish evaluating; iter over results'''
        keys = list(map(self._encode, to_read))
        futures = [self.pending_evals[key] for key in keys]
        logger.info(f"Waiting on {len(keys)} evals to finish...")
        result = concurrent.futures.wait(futures)
        for x, key, future in zip(to_read, keys, futures):
            y = future.result()
            self.evals[key] = y
            del self.pending_evals[key]
            logger.info(f"x: {x} y: {y}")
            yield (x, y)
        
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

    def __getstate__(self):
        d = {}
        d['evals'] = self.evals
        d['pending_evals'] = list(self.pending_evals.keys())
        d['backend'] = self.backend
        d['executor'] = None
        d['num_workers'] = self.num_workers
        d['params_list'] = self.params_list
        d['bench_module_name'] = self.bench_module_name
        return d

    def __setstate__(self, d):
        logger.info(f"Unpickling LocalEvaluator")

        self.evals = d['evals']
        self.pending_evals = {}
        self.backend = d['backend']
        self.executor = d['executor']
        self.num_workers = d['num_workers']
        self.params_list = d['params_list']
        self.bench_module_name = d['bench_module_name']
        
        self.bench_module = import_module(self.bench_module_name)
        pending_eval_keys = d['pending_evals']
        
        logger.info(f"Restored {len(self.evals)} finished evals")
        logger.info(f"Resuming {len(pending_eval_keys)} evals")

        for key in pending_eval_keys:
            x = self._decode(key)
            self.add_eval(x)

import concurrent.futures
import logging
from deephyper.evaluators import evaluate
from importlib import import_module
import os

logger = logging.getLogger(__name__)

class LocalEvaluator(evaluate.Evaluator):
    ExecutorCls = concurrent.futures.ProcessPoolExecutor

    def __init__(self, params_list, bench_module_name, num_workers=None,
                 backend='tensorflow', model_path='', data_source='', 
                 stage_in_destination=''):
        super().__init__()
        self.executor = None
        self.num_workers = num_workers
        self.backend = backend
        self.params_list = params_list
        self.bench_module_name = bench_module_name
        self.bench_module = import_module(bench_module_name)
        self.model_path = model_path
        self.data_source = data_source
        self.stage_in_destination = stage_in_destination
        logger.info("Local Evaluator instantiated")
        logger.info(f"Backend: {self.backend}")
        logger.info(f"Benchmark: {bench_module_name}")
        self._setup_executor()

    def _setup_executor(self):
        os.environ['KERAS_BACKEND'] = self.backend
        self.executor = self.ExecutorCls(max_workers=self.num_workers)
        self.num_workers = self.executor._max_workers
        logger.info(f"Created ProcessPool with {self.num_workers} procs")

    def _eval_exec(self, x):
        if self.executor is None:
            self._setup_executor()

        param_dict = {k:v for k,v in zip(self.params_list, x) if 'hidden' not in k}
        param_dict['model_path'] = self.model_path
        param_dict['data_source'] = self.data_source
        param_dict['stage_in_destination'] = self.stage_in_destination
        eval_func = self.bench_module.run
        future = self.executor.submit(eval_func, param_dict)
        logger.info(f"Running: {param_dict}")
        return future

    def await_evals(self, to_read):
        '''wait for a set of points to finish evaluating; iter over results'''
        keys = list(map(self._encode, to_read))

        results = []
        for x, key in zip(to_read, keys):
            if key in self.pending_evals:
                future = self.pending_evals[key]
                results.append( (x, key, future) )
            else:
                y = self.evals[key]
                logger.info(f"Already evaluated x: {x} y: {y}")
                results.append( (x, key, y) )

        futures = [r[2] for r in results if isinstance(r, concurrent.futures.Future)]
        logger.info(f"Waiting on {len(futures)} evals to finish...")
        done = concurrent.futures.wait(futures)

        for (x, key, future) in results:
            if isinstance(future, concurrent.futures.Future):
                y = future.result()
            else:
                y = future
            self.evals[key] = y
            if key in self.pending_evals: del self.pending_evals[key]
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
        d['num_workers'] = self.num_workers
        d['backend'] = self.backend
        d['params_list'] = self.params_list
        d['bench_module_name'] = self.bench_module_name
        d['pending_evals'] = list(self.pending_evals.keys())
        d['evals'] = self.evals
        d['executor'] = None
        d['bench_module'] = None
        return d

    def stop(self):
        for future in self.pending_evals:
            future.cancel()
        self.executor.shutdown()

    def __setstate__(self, d):
        logger.info(f"Unpickling LocalEvaluator")

        self.__dict__ = d
        self.bench_module = import_module(self.bench_module_name)

        self.pending_evals = {}
        pending_eval_keys = d['pending_evals']
        
        logger.info(f"Restored {len(self.evals)} finished evals")
        logger.info(f"Resuming {len(pending_eval_keys)} evals")

        for key in pending_eval_keys:
            x = self._decode(key)
            self.add_eval(x)

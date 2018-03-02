from numpy import integer, floating, ndarray
import json
import uuid
import multiprocessing
import concurrent.futures

class Encoder(json.JSONEncoder):
    '''Enables JSON dump of numpy data'''
    def default(self, obj):
        if isinstance(obj, uuid.UUID): return obj.hex
        if isinstance(obj, integer): return int(obj)
        elif isinstance(obj, floating): return float(obj)
        elif isinstance(obj, ndarray): return obj.tolist()
        else: return super(Encoder, self).default(obj)

class Evaluator:

    def __init__(self):
        self.points = {}
        self.already_read = []

    def add_point(x, cfg):
        x_id = self._new_point(x, cfg)
        self.points[str(x_id)] = json.dumps(x, cls=Encoder)
        self._submit_eval(x, cfg)

    def read_points(self):
        raise NotImplementedError

    def _submit_eval(self):
        raise NotImplementedError

    @property
    def counter(self):
        return len(self.points)

class LocalEvaluator(Evaluator):
    ExecutorCls = concurrent.futures.ProcessPoolExecutor

    def __init__(self):
        super(self).__init__()
        self.executor = None
        self.evals = []

    def _submit_eval(self, x, cfg):
        if self.executor is None:
            os.environ['KERAS_BACKEND'] = cfg.backend
            self.executor = ExecutorCls()

        kwargs = {k:v for k,v in zip(cfg.params, x) if 'hidden' not in k}
        eval_func = cfg.benchmark_module.run
        future = self.executor.submit(eval_func, kwargs=kwargs)
        self.evals.append((x, future))

    def read_points(self):
        for (x, future) in self.futures:
            if future.done():

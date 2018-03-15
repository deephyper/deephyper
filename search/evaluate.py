from numpy import integer, floating, ndarray
import json
import uuid
import multiprocessing
import logging

logger = logging.getLogger(__name__)

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
        self.pending_evals = {} # x --> future or x --> UUID
        self.evals = {} # x --> cost

    def encode(self, x):
        return self._encode(x)

    def _encode(self, x):
        '''from x (list) to JSON string'''
        return json.dumps(x, cls=Encoder)

    def _decode(self, key):
        '''from JSON string to x (list)'''
        return json.loads(key)

    def add_eval(self, x, re_evaluate=False):
        key = self._encode(x)
        if key in self.evals or key in self.pending_evals:
            if not re_evaluate:
                logger.info(f"Point {key} has already been evaluated! Skipping.")
                return
        new_eval = self._eval_exec(x) # future or job UUID
        self.pending_evals[key] = new_eval

    @property
    def counter(self):
        return len(self.evals) + len(self.pending_evals)
    
    def num_free_workers(self):
        num_evals = len(self.pending_evals)
        logger.debug(f"{num_evals} pending evals; {self.num_workers} workers")
        if num_evals <= self.num_workers:
            return self.num_workers - num_evals
        else:
            return 0

    def dump_evals(self):
        with open('results.json', 'w') as fp:
            json.dump(self.evals, fp, indent=4, sort_keys=True, cls=Encoder)

        resultsList = []

        for key in self.evals:
            x = self._decode(key)
            resultDict = {name : value for (name,value) 
                          in zip(self.params_list, x)}
            resultDict['objective'] = self.evals[key]
            resultsList.append(resultDict)

        with open('results.csv', 'w') as fp:
            writer = csv.DictWriter(fp, keys)
            writer.writeheader()
            writer.writerows(resultsList)


def create_evaluator(opt_config):
    evaluator_class = opt_config.evaluator
    assert evaluator_class in ['balsam', 'local']

    if evaluator_class == "balsam":
        from deephyper.search.evaluate_balsam import BalsamEvaluator
        cls = BalsamEvaluator
    else:
        from deephyper.search.evaluate_local import LocalEvaluator
        cls = LocalEvaluator

    evaluator = cls(opt_config.params,
                    opt_config.benchmark_module_name,
                    num_workers=opt_config.num_workers,
                    backend=opt_config.backend
                   )
    return evaluator

import json
import logging
import os
import pickle
import signal
import sys
import numpy as np
from numpy import integer, floating, ndarray

HERE = os.path.dirname(os.path.abspath(__file__)) # search dir
top  = os.path.dirname(os.path.dirname(HERE)) # directory containing deephyper
sys.path.append(top)

from deephyper.evaluators import evaluate
from deephyper.search import util


logger = util.conf_logger('deephyper.search.amls')

SERVICE_PERIOD = 2          # Delay (seconds) between main loop iterations
CHECKPOINT_INTERVAL = 10    # How many jobs to complete between optimizer checkpoints
SEED = 12345

profile_timer = util.Timer()

def save_checkpoint(opt_config, optimizer, evaluator):
    if evaluator.counter == 0: return
    data = {}
    data['opt_config'] = opt_config
    data['optimizer'] = optimizer
    data['evaluator'] = evaluator

    if evaluator.evals:
        best = min(evaluator.evals.items(), key=lambda x: x[1])
        data['best'] = best
        logger.info(f'best point: {best}')
    
    fname = f'{opt_config.benchmark}.pkl'
    with open(fname, 'wb') as fp:
        pickle.dump(data, fp)

    evaluator.dump_evals()
    logger.info(f"Checkpointed run in {os.path.abspath(fname)}")

def load_checkpoint(chk_path):
    chk_path = os.path.abspath(os.path.expanduser(chk_path))
    assert os.path.exists(chk_path), "No such checkpoint file"
    with open(chk_path, 'rb') as fp: data = pickle.load(fp)
    
    cfg, opt, evaluator = data['opt_config'], data['optimizer'], data['evaluator']

    cfg.num_workers = args.num_workers
    logger.info(f"Resuming from checkpoint in {chk_path}")
    logger.info(f"On eval {evaluator.counter}")
    return cfg, opt, evaluator

class Optimizer:
    class Encoder(json.JSONEncoder):
        '''JSON dump of numpy data'''
        def default(self, obj):
            if isinstance(obj, integer): return int(obj)
            elif isinstance(obj, floating): return float(obj)
            elif isinstance(obj, ndarray): return obj.tolist()
            else: return super(Encoder, self).default(obj)
    
    def encode(self, x):
        return json.dumps(x, cls=self.Encoder)

    def decode(self, key):
        return json.loads(key)

    def __init__(self, cfg):
        self._optimizer = util.sk_optimizer_from_config(cfg, SEED)
        assert cfg.amls_lie_strategy in "cl_min cl_mean cl_max".split()
        self.strategy = cfg.amls_lie_strategy
        self.evals = {}

    def _get_lie(self):
        if self.strategy == "cl_min":
            return min(self._optimizer.yi) if self._optimizer.yi else 0.0
        elif self.strategy == "cl_mean":
            return self._optimizer.yi.mean() if self._optimizer.yi else 0.0
        else:
            return  max(self._optimizer.yi) if self._optimizer.yi else 0.0

    def xy_from_dict(self):
        keys = list(self.evals.keys())
        XX = [self.decode(x) for x in keys]
        YY = [self.evals[x] for x in keys]
        return XX, YY

    def _ask(self):
        x = self._optimizer.ask()
        y = self._get_lie()
        self._optimizer.tell(x,y)
        self.evals[self.encode(x)] = y
        return x

    def ask(self, n_points=None):
        if n_points is None:
            return self._ask()
        else:
            return [self._ask() for i in range(n_points)]
        
    def tell(self, xy_data):
        assert isinstance(xy_data, list)
        maxval = max(self._optimizer.yi) if self._optimizer.yi else 0.0
        for x,y in xy_data:
            self.evals[self.encode(x)] = (y if y < sys.float_info.max else maxval)

        XX, YY = self.xy_from_dict()
        self._optimizer.Xi = []
        self._optimizer.yi = []
        self._optimizer.tell(XX, YY)

def main(args):
    '''Service loop: add jobs; read results; drive optimizer'''

    # Initialize optimizer
    if args.from_checkpoint:
        chk_path = args.from_checkpoint
        cfg, optimizer, evaluator = load_checkpoint(chk_path)
    else:
        cfg = util.OptConfig(args)
        optimizer = Optimizer(cfg)
        evaluator = evaluate.create_evaluator(cfg)
        logger.info(f"Starting new run with {cfg.benchmark_module_name}")

    timer = util.DelayTimer(max_minutes=None, period=SERVICE_PERIOD)
    chkpoint_counter = 0

    # Gracefully handle shutdown
    def handler(signum, stack):
        evaluator.stop()
        logger.info('Received SIGINT/SIGTERM')
        save_checkpoint(cfg, optimizer, evaluator)
        sys.exit(0)

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)

    # INITIAL POINTS
    logger.info("AMLS-Balsam driver starting")
    logger.info(f"Generating {cfg.num_workers} initial points...")
    XX = optimizer._optimizer.ask(n_points=cfg.num_workers)
    for x in XX: evaluator.add_eval(x, re_evaluate=cfg.repeat_evals)

    # MAIN LOOP
    for elapsed_str in timer:
        logger.info(f"Elapsed time: {elapsed_str}")
        if len(evaluator.evals) == cfg.max_evals: break

        results = list(evaluator.get_finished_evals())
        if results:
            logger.info(f"Refitting model with batch of {len(results)} evals")
            optimizer.tell(results)
            logger.info(f"Drawing {len(results)} points with strategy {optimizer.strategy}")
            XX = optimizer.ask(n_points=len(results))
            for x in XX: evaluator.add_eval(x, re_evaluate=cfg.repeat_evals)
            chkpoint_counter += len(results)

        if chkpoint_counter >= CHECKPOINT_INTERVAL:
            save_checkpoint(cfg, optimizer, evaluator)
            chkpoint_counter = 0
        sys.stdout.flush()
    
    # EXIT
    logger.info('Hyperopt driver finishing')
    save_checkpoint(cfg, optimizer, evaluator)

if __name__ == "__main__":
    parser = util.create_parser()
    args = parser.parse_args()
    main(args)

import json
import logging
import os
import pickle
import signal
import sys
import numpy as np
from numpy import integer, floating, ndarray

from deephyper.search import Search

HERE = os.path.dirname(os.path.abspath(__file__)) # search dir
top  = os.path.dirname(os.path.dirname(HERE)) # directory containing deephyper
sys.path.append(top)

from deephyper.evaluators import Evaluator
from deephyper.search import util

logger = util.conf_logger('deephyper.search.ambs')

SERVICE_PERIOD = 2          # Delay (seconds) between main loop iterations
CHECKPOINT_INTERVAL = 10    # How many jobs to complete between optimizer checkpoints
SEED = 12345

profile_timer = util.Timer()

def save_checkpoint(args, optimizer, evaluator):
    if evaluator.counter == 0: return
    data = {}
    data['args'] = args
    data['optimizer'] = optimizer
    data['evaluator'] = evaluator

    if evaluator.evals:
        best = min(evaluator.evals.items(), key=lambda x: x[1])
        data['best'] = best
        logger.info(f'best point: {best}')

    fname = f'{args.problem}.pkl'
    with open(fname, 'wb') as fp:
        pickle.dump(data, fp)

    evaluator.dump_evals()
    logger.info(f"Checkpointed run in {os.path.abspath(fname)}")

def load_checkpoint(chk_path):
    chk_path = os.path.abspath(os.path.expanduser(chk_path))
    assert os.path.exists(chk_path), "No such checkpoint file"
    with open(chk_path, 'rb') as fp: data = pickle.load(fp)

    args, opt, evaluator = data['args'], data['optimizer'], data['evaluator']

    args.num_workers = args.num_workers
    logger.info(f"Resuming from checkpoint in {chk_path}")
    logger.info(f"On eval {evaluator.counter}")
    return args, opt, evaluator

class Optimizer:
    def __init__(self, problem, num_workers, args):
        self._optimizer = util.sk_optimizer_from_config(problem, num_workers, args, SEED)
        assert args.ambs_lie_strategy in "cl_min cl_mean cl_max".split()
        self.strategy = args.ambs_lie_strategy
        self.evals = {}
        self.counter = 0

    def _get_lie(self):
        if self.strategy == "cl_min":
            return min(self._optimizer.yi) if self._optimizer.yi else 0.0
        elif self.strategy == "cl_mean":
            return self._optimizer.yi.mean() if self._optimizer.yi else 0.0
        else:
            return  max(self._optimizer.yi) if self._optimizer.yi else 0.0

    def _xy_from_dict(self):
        keys = list(self.evals.keys())
        XX = [x for x in keys]
        YY = [self.evals[x] for x in keys]
        return XX, YY

    def _ask(self):
        x = self._optimizer.ask()
        y = self._get_lie()
        self._optimizer.tell(x,y)
        self.evals[x] = y
        return x

    def ask(self, n_points=None, batch_size=20):
        if n_points is None:
            self.counter += 1
            return self._ask()
        else:
            self.counter += n_points
            batch = []
            for i in range(n_points):
                batch.append(self._ask())
                if len(batch) == batch_size:
                    yield batch
                    batch = []
            if batch:
                yield batch

    def ask_initial(self, n_points):
        XX = self._optimizer.ask(n_points=n_points)
        for x in XX:
            key = x
            self.evals[key] = 0.0
        self.counter += n_points
        return XX

    def tell(self, xy_data):
        assert isinstance(xy_data, list)
        maxval = max(self._optimizer.yi) if self._optimizer.yi else 0.0
        for x,y in xy_data:
            key = x
            assert key in self.evals
            self.evals[key] = (y if y < sys.float_info.max else maxval)

        self._optimizer.Xi = []
        self._optimizer.yi = []
        XX, YY = self._xy_from_dict()
        assert len(XX) == len(YY) == self.counter
        self._optimizer.tell(XX, YY)
        assert len(self._optimizer.Xi) == len(self._optimizer.yi) == self.counter

def key(d):
    return json.dumps(d)

class AMBS(Search):
    def __init__(self, args):
        """Service loop: add jobs; read results; drive optimizer
        """
        self.args = args
        self.problem = util.load_attr_from(self.args.problem)()
        run_func = util.load_attr_from(self.args.run)
        self.evaluator = Evaluator.create(run_func, cache_key=key, method=args.evaluator)
        self.num_workers = self.evaluator.num_workers
        self.optimizer = Optimizer(self.problem, self.num_workers, self.args)


    def _handler(self, signum, stack):
        self.evaluator.stop()
        logger.info('Received SIGINT/SIGTERM')
        save_checkpoint(self.args, self.optimizer, self.evaluator)
        sys.exit(0)

    def run(self):

        timer = util.DelayTimer(max_minutes=None, period=SERVICE_PERIOD)
        chkpoint_counter = 0

        # Gracefully handle shutdown
        signal.signal(signal.SIGINT, self._handler)
        signal.signal(signal.SIGTERM, self._handler)

        # INITIAL POINTS
        logger.info("AMLS-single server driver starting")
        logger.info(f"Generating {self.args.num_workers} initial points...")
        XX = self.optimizer.ask_initial(n_points=self.num_workers)
        for x in XX: self.evaluator.add_eval(x)

        # MAIN LOOP
        num_evals = 0
        for elapsed_str in timer:
            logger.info(f"Elapsed time: {elapsed_str}")
            if len(num_evals) == self.args.max_evals: break

            results = list(self.evaluator.get_finished_evals())
            if results:
                num_evals += len(results)
                logger.info(f"Refitting model with batch of {len(results)} evals")

                self.optimizer.tell(results)

                logger.info(f"Drawing {len(results)} points with strategy {self.optimizer.strategy}")
                for batch in self.optimizer.ask(n_points=len(results)):
                    self.evaluator.add_eval_batch(batch, re_evaluate=self.args.repeat_evals)
                chkpoint_counter += len(results)

            if chkpoint_counter >= CHECKPOINT_INTERVAL:
                save_checkpoint(self.args, self.optimizer, self.evaluator)
                chkpoint_counter = 0
            sys.stdout.flush()

        # EXIT
        logger.info('Hyperopt driver finishing')
        save_checkpoint(self.args, self.optimizer, self.evaluator)

if __name__ == "__main__":
    parser = util.create_parser()
    args = parser.parse_args()
    search = AMBS(args)
    search.run()

import numpy as np
import glob

from random import random
from math import log, ceil
from time import time, ctime

import logging
import os
import pickle
import signal
import sys

HERE = os.path.dirname(os.path.abspath(__file__)) # search dir
top  = os.path.dirname(os.path.dirname(HERE)) # directory containing deephyper
sys.path.append(top)

from deephyper.evaluators import evaluate
from deephyper.search import util

logger = util.conf_logger('deephyper.search.hyperband')

SERVICE_PERIOD = 2          # Delay (seconds) between main loop iterations
CHECKPOINT_INTERVAL = 30    # How many jobs to complete between optimizer checkpoints
SEED = 12345

class Hyperband:
    def __init__(self, cfg, evaluator, eval_timeout_minutes):
        self.evaluator = evaluator
        self.opt_config = cfg
        self.eval_timeout_seconds = eval_timeout_minutes * 60
        self.opt_config.learner = "DUMMY"
        self.optimizer = util.sk_optimizer_from_config(self.opt_config, SEED)

        self.max_iter = self.opt_config.num_workers # maximum iterations per configuration
        self.eta = 3			# defines configuration downsampling rate (default = 3)

        self.logeta = lambda x: log( x ) / log( self.eta )
        self.s_max = int(self.logeta(self.max_iter))
        self.B = ( self.s_max + 1 ) * self.max_iter

        self.best_loss = np.inf

    def save_checkpoint(self):
        self.evaluator.dump_evals()

    def clear_old_models(self):
        path = self.opt_config.model_path
        path = os.path.abspath(os.path.expanduser(path))
        print("Clearing all .h5 and .pkl files in {path}")
        pattern = os.path.join(path, '*.h5')
        for fname in glob.glob(pattern): os.remove(fname)
        pattern = os.path.join(path, '*.pkl')
        for fname in glob.glob(pattern): os.remove(fname)

    def run(self):
        self.clear_old_models()
        for s in reversed( range( self.s_max + 1 )):
            n = int( ceil( self.B / self.max_iter / ( s + 1 ) * self.eta ** s ))
            r = round(self.max_iter * self.eta ** ( -s ))

            if self.opt_config.starting_point is not None:
                T = [self.opt_config.starting_point]
                self.opt_config.starting_point = None
                additional_pts = self.optimizer.ask(n_points=n-1)
                T.extend(additional_pts)
            else:
                T = self.optimizer.ask(n_points=n)

            self._inner_loop(s, n, r, T)
            self.clear_old_models()
        print("Hyperband run done")

    def _inner_loop(self, s, n, r, T):
        epochs_index = self.opt_config.params.index('epochs')
        print(f"Hyperband Inner Loop n={n} r={r}")

        for i in range(( s + 1 )):
            # Run each of the n_configs for <n_iterations>
            n_configs = len(T)
            n_iterations = round(r * self.eta **i)
            
            for t in T: t[epochs_index] = n_iterations
            assert all(t[epochs_index] == n_iterations for t in T)
            print(f'==> n_configs={n_configs} n_iterations={n_iterations}')

            for t in T: self.evaluator.add_eval(t, re_evaluate=True)
            eval_results = self.evaluator.await_evals(T, timeout_sec=self.eval_timeout_seconds) # barrier
            val_losses = [loss for (t, loss) in eval_results]
            assert len(val_losses) == len(T)

            best_t, best_loss = min(self.evaluator.evals.items(), key=lambda x: x[1])
            if best_loss < self.best_loss:
                print("best loss so far:", best_loss)
                print(best_t, '\n')
                self.best_loss = best_loss

            # keep best (n_configs / eta) configurations (TODO: filter out early stops)
            indices = np.argsort( val_losses )
            T = [T[i] for i in indices]
            T = T[:int(n_configs / self.eta)]
            sys.stdout.flush()
            self.save_checkpoint()


def main(args):
    '''Service loop: add jobs; read results; drive optimizer'''

    cfg = util.OptConfig(args)
    evaluator = evaluate.create_evaluator(cfg)
    hyperband = Hyperband(cfg, evaluator, args.eval_timeout_minutes)
    logger.info(f"Starting Hyperband run with {cfg.benchmark_module_name}")

    timer = util.DelayTimer(max_minutes=None, period=SERVICE_PERIOD)

    # Gracefully handle shutdown
    def handler(signum, stack):
        evaluator.stop()
        logger.info('Received SIGINT/SIGTERM')
        hyperband.save_checkpoint()
        sys.exit(0)

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)

    for time_str in timer:
        logger.debug(f"Starting hyperband run at elapsed time {time_str}")
        hyperband.run()

if __name__ == "__main__":
    parser = util.create_parser()
    args = parser.parse_args()
    if not args.model_path:
        raise ValueError("Need to specify model_path directory where "
        "intermediate models will be loaded/stored")
    main(args)

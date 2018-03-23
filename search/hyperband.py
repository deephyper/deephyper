import numpy as np

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

masterLogger = util.conf_logger()
logger = logging.getLogger(__name__)

SERVICE_PERIOD = 2          # Delay (seconds) between main loop iterations
CHECKPOINT_INTERVAL = 30    # How many jobs to complete between optimizer checkpoints
SEED = 12345

def save_checkpoint(opt_config, optimizer, evaluator):
    data = {}
    data['opt_config'] = opt_config
    data['optimizer'] = optimizer
    data['evaluator'] = evaluator
    
    fname = f'{opt_config.benchmark}.pkl'
    with open(fname, 'wb') as fp:
        pickle.dump(data, fp)

    evaluator.dump_evals()
    logger.info(f"Checkpointed run in {os.path.abspath(fname)}")

class Hyperband:
    def __init__(self, cfg, evaluator):
        self.evaluator = evaluator
        self.opt_config = cfg
        self.opt_config.learner = "DUMMY"
        self.optimizer = util.sk_optimizer_from_config(self.opt_config, SEED)

        self.max_iter = self.opt_config.num_workers # maximum iterations per configuration
        self.eta = 3			# defines configuration downsampling rate (default = 3)

        self.logeta = lambda x: log( x ) / log( self.eta )
        self.s_max = int(self.logeta(self.max_iter))
        self.B = ( self.s_max + 1 ) * self.max_iter

        self.best_loss = np.inf

    def run(self):
        for s in reversed( range( self.s_max + 1 )):
            n = int( ceil( self.B / self.max_iter / ( s + 1 ) * self.eta ** s ))
            r = self.max_iter * self.eta ** ( -s )

            if self.opt_config.starting_point is not None:
                T = [self.opt_config.starting_point]
                self.opt_config.starting_point = None
                additional_pts = self.optimizer.ask(n_points=n-1)
                T.extend(additional_pts)
            else:
                T = self.optimizer.ask(n_points=n)

            self._inner_loop(s, n, r, T)

    def _inner_loop(self, s, n, r, T):
        epochs_index = self.opt_config.params.index('epochs')
        print(f"Hyperband Inner Loop n={n} r={r}")

        for i in range(( s + 1 )):
            # Run each of the n_configs for <n_iterations>
            n_configs = len(T)
            n_iterations = r * self.eta ** ( i )
            
            for t in T: t[epochs_index] = n_iterations
            assert all(t[epochs_index] == n_iterations for t in T)
            print(f'==> n_configs={n_configs} n_iterations={n_iterations}')

            for t in T: self.evaluator.add_eval(t, re_evaluate=True)
            eval_results = self.evaluator.await_evals(T) # barrier
            val_losses = [loss for (t, loss) in eval_results]

            best_t, best_loss = min(evaluator.evals.items(), key=lambda x: x[1])
            if best_loss < self.best_loss:
                print("best loss so far:", best_loss)
                print(best_t, '\n')
                self.best_loss = best_loss

            # keep best (n_configs / eta) configurations (TODO: filter out early stops)
            indices = np.argsort( val_losses )
            T = [T[i] for i in indices]
            T = T[:int(n_configs / self.eta)]


def main(args):
    '''Service loop: add jobs; read results; drive optimizer'''

    # Initialize optimizer
    if args.from_checkpoint:
        chk_path = args.from_checkpoint
        cfg, optimizer, evaluator = load_checkpoint(chk_path)
    else:
        cfg = util.OptConfig(args)
        optimizer = Hyperband(cfg)

        evaluator = evaluate.create_evaluator(cfg)
        logger.info(f"Starting new run with {cfg.benchmark_module_name}")

    timer = util.elapsed_timer(max_runtime_minutes=None, service_period=SERVICE_PERIOD)
    chkpoint_counter = 0

    # Gracefully handle shutdown
    def handler(signum, stack):
        evaluator.stop()
        logger.info('Received SIGINT/SIGTERM')
        save_checkpoint(cfg, optimizer, evaluator)
        sys.exit(0)

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)

    # MAIN LOOP
    logger.info("Hyperopt driver starting")

    for elapsed_seconds in timer:
        logger.info("\nElapsed time:", util.pretty_time(elapsed_seconds))
        hyperband = Hyperband(cfg, optimizer, evaluator)
        results = hyperband.run()

        save_checkpoint(cfg, optimizer, evaluator)
        sys.stdout.flush()

    # EXIT
    logger.info('Hyperopt driver finishing')
    save_checkpoint(cfg, optimizer, evaluator)

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method('forkserver')
    parser = util.create_parser()
    args = parser.parse_args()
    if not args.model_path:
        raise ValueError("Need to specify model_path directory where "
        "intermediate models will be loaded/stored")
    main(args)

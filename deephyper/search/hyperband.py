import os
import glob
import sys
import signal
import numpy as np

from math import log, ceil

from deephyper.search.optimizers import Optimizer
from deephyper.search import Search
from deephyper.search import util

logger = util.conf_logger('deephyper.search.hyperband')

SERVICE_PERIOD = 2          # Delay (seconds) between main loop iterations
CHECKPOINT_INTERVAL = 10    # How many jobs to complete between optimizer checkpoints
EXIT_FLAG = False

def on_exit(signum, stack):
    global EXIT_FLAG
    EXIT_FLAG = True

class Hyperband(Search):
    def __init__(self):
        super().__init__()
        logger.info("Initializing Hyperband")
        self.optimizer = Optimizer(self.problem, self.num_workers, self.args)

        self.max_iter = self.num_workers # maximum iterations per configuration
        self.eta = 3 # defines configuration downsampling rate (default = 3)

        self.logeta = lambda x: log( x ) / log( self.eta )
        self.s_max = int(self.logeta(self.max_iter))
        self.B = ( self.s_max + 1 ) * self.max_iter

        self.best_loss = np.inf

    def _extend_parser(self, parser):
        parser.add_argument('--learner',
            default='RF',
            choices=["RF", "ET", "GBRT", "DUMMY", "GP"],
            help='type of learner (surrogate model)'
        )
        parser.add_argument('--acq-func',
            default="gp_hedge",
            choices=["LCB", "EI", "PI","gp_hedge"],
            help='Acquisition function type'
        )
        parser.add_argument('--liar-strategy',
            default="cl_max",
            choices=["cl_min", "cl_mean", "cl_max"],
            help='Constant liar strategy'
        )
        return parser

    def run(self):
        for s in reversed( range( self.s_max + 1 )):
            n = int( ceil( self.B / self.max_iter / ( s + 1 ) * self.eta ** s ))
            r = round(self.max_iter * self.eta ** ( -s ))

            if self.problem.starting_point is not None:
                T = [self.problem.starting_point]
                self.problem.starting_point = None
                additional_pts = self.optimizer.ask(n_points=n-1)
                T.extend(additional_pts)
            else:
                T = self.optimizer.ask(n_points=n)

            self._inner_loop(s, n, r, T)
        print("Hyperband run done")

    def _inner_loop(self, s, n, r, T):
        print(f'type: {type(self.problem.params)}')
        epochs_index = self.problem.params.index('epochs')
        print('LA')
        print(f"Hyperband Inner Loop n={n} r={r}")

        for i in range(( s + 1 )):
            # Run each of the n_configs for <n_iterations>
            n_configs = len(T)
            n_iterations = round(r * self.eta **i)

            for t in T: t[epochs_index] = n_iterations
            assert all(t[epochs_index] == n_iterations for t in T)
            print(f'==> n_configs={n_configs} n_iterations={n_iterations}')

            for t in T: self.evaluator.add_eval(t, re_evaluate=True)
            eval_results = self.evaluator.await_evals(T, timeout_sec=self.args.eval_timeout_seconds) # barrier
            val_losses = [loss for (t, loss) in eval_results]
            assert len(val_losses) == len(T)

            best_t, best_loss = min(self.evaluator.finished_evals.items(), key=lambda x: x[1])
            if best_loss < self.best_loss:
                print("best loss so far:", best_loss)
                print(best_t, '\n')
                self.best_loss = best_loss

            # keep best (n_configs / eta) configurations (TODO: filter out early stops)
            indices = np.argsort( val_losses )
            T = [T[i] for i in indices]
            T = T[:int(n_configs / self.eta)]
            sys.stdout.flush()

if __name__ == "__main__":
    search = Hyperband()
    signal.signal(signal.SIGINT, on_exit)
    signal.signal(signal.SIGTERM, on_exit)
    search.run()

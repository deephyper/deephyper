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

from skopt import Optimizer

masterLogger = util.conf_logger()
logger = logging.getLogger(__name__)

SERVICE_PERIOD = 2          # Delay (seconds) between main loop iterations
CHECKPOINT_INTERVAL = 30    # How many jobs to complete between optimizer checkpoints
SEED = 12345

def evaluate_fitnesses(points, opt, evaluator):
    for x in points: evaluator.add_eval(x)
    logger.info(f"Waiting on {len(points)} individual fitness evaluations")
    results = evaluator.await_evals(points)
    return list(results)

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

    def __init__(self, cfg, optimizer, evaluator ):
        self.optimizer = optimizer
        self.evaluator = evaluator
        self.opt_config = cfg

        self.max_iter = 81		# maximum iterations per configuration
        self.eta = 3			# defines configuration downsampling rate (default = 3)

        self.logeta = lambda x: log( x ) / log( self.eta )
        self.s_max = int( self.logeta( self.max_iter ))
        self.B = ( self.s_max + 1 ) * self.max_iter

        self.results = []	# list of dicts
        self.counter = 0
        self.best_loss = np.inf
        self.best_counter = -1
        # can be called multiple times
        def run( self, skip_last = 0, dry_run = False ):
            for s in reversed( range( self.s_max + 1 )):
                # initial number of configurations
                n = int( ceil( self.B / self.max_iter / ( s + 1 ) * self.eta ** s ))	
                # initial number of iterations per config
                r = self.max_iter * self.eta ** ( -s )		
                # n random configurations
                #T = [ self.get_params() for i in range( n )]
                if self.opt_config.starting_point is not None:
                    T = [self.opt_config.starting_point]
                    self.opt_config.starting_point = None
                    additional_pts = self.optimizer.ask(n_points=n-1)
                    T.extend(additional_pts)
                else:
                    T = self.optimizer.ask(n_points=n)
                    for i in range(( s + 1 )):	# changed from s + 1
                        print('==> (%d, %d, %d) ' % (s, len(T), r))
                        # Run each of the n configs for <iterations> 
                        # and keep best (n_configs / eta) configurations
                        n_configs = n * self.eta ** ( -i )
                        n_iterations = r * self.eta ** ( i )
                        #print "\n*** {} configurations x {:.1f} iterations each".format(n_configs, n_iterations )
                        val_losses = []
                        early_stops = []
                        for t in T:		
                            self.counter += 1
                            self.evaluator.add_eval(t, re_evaluate=True)
                            results = self.evaluator.await_evals(T) # barrier

                            for (t, loss) in results:
                                result = {}
                                result['loss'] = loss	
                                val_losses.append(loss)
                                early_stops.append(False)
                                # early_stop = result.get('early_stop', False)
                                # early_stops.append(early_stop)
                                # keeping track of the best result so far (for display only)
                                # could do it be checking results each time, but hey
                                if loss < self.best_loss:
                                    self.best_loss = loss
                                    self.best_counter = self.counter
                                    result['counter'] = self.counter
                                    #result['seconds'] = seconds
                                    result['params'] = t
                                    result['iterations'] = n_iterations
                                    self.results.append(result)

                                # select a number of best configurations for the next loop
                                # filter out early stops, if any
                                indices = np.argsort( val_losses )
                                T = [ T[i] for i in indices if not early_stops[i]]
                                T = T[ 0:int( n_configs / self.eta )]

                return self.results

def main(args):
    '''Service loop: add jobs; read results; drive optimizer'''

    # Initialize optimizer
    if args.from_checkpoint:
        chk_path = args.from_checkpoint
        cfg, optimizer, evaluator = load_checkpoint(chk_path)
    else:
        cfg = util.OptConfig(args)

        optimizer = Optimizer(
            cfg.space,
            base_estimator='dummy',
            acq_optimizer='sampling',
            n_initial_points=np.inf,
            random_state=SEED)
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
    main(args)

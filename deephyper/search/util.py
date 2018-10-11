import argparse
import os
import csv
import time
import logging
from importlib import import_module
from importlib.util import find_spec
from pprint import pprint
import numpy as np

masterLogger = None

class Timer:
    def __init__(self):
        self.timings = {}

    def start(self, name):
        self.timings[name] = time.time()

    def end(self, name):
        try:
            elapsed = time.time() - self.timings[name]
        except KeyError:
            print(f"TIMER error: never called timer.start({name})")
        else:
            print(f"TIMER {name}: {elapsed:.4f} seconds")
            del self.timings[name]


def sk_optimizer_from_config(problem, num_workers, args, random_state):
    from skopt import Optimizer
    from numpy import inf
    logger = logging.getLogger(__name__)
    kappa = 1.96
    space = list(problem.space.values())

    if args.learner in "RF ET GBRT GP".split():
        n_init = num_workers
    else:
        assert args.learner == "DUMMY"
        n_init = inf

    if args.learner in ["RF", "ET", "GBRT", "GP", "DUMMY"]:
        optimizer = Optimizer(
            space,
            base_estimator=args.learner,
            acq_optimizer='sampling',
            acq_func=args.ambs_acq_func,
            acq_func_kwargs={'kappa':kappa},
            random_state=random_state,
            n_initial_points=n_init
        )
    else:
        raise ValueError(f"Unknown learner type {args.learner}")
    logger.info("Creating skopt.Optimizer with %s base_estimator" % args.learner)
    return optimizer

def conf_logger(name):
    global masterLogger
    if (masterLogger == None):
        masterLogger = logging.getLogger('deephyper')

        handler = logging.FileHandler('deephyper.log')
        formatter = logging.Formatter(
            '%(asctime)s|%(process)d|%(levelname)s|%(name)s:%(lineno)s] %(message)s',
            "%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        masterLogger.addHandler(handler)
        masterLogger.setLevel(logging.DEBUG)
        masterLogger.info("\n\nLoading Deephyper\n--------------")
    return logging.getLogger(name)

class DelayTimer:
    def __init__(self, max_minutes=None, period=2):
        if max_minutes is None:
            max_minutes = float('inf')
        self.max_minutes = max_minutes
        self.max_seconds = max_minutes * 60.0
        self.period = period
        self.delay = True

    def pretty_time(self, seconds):
        """Format time string"""
        seconds = round(seconds, 2)
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        return "%02d:%02d:%02.2f" % (hours,minutes,seconds)

    def __iter__(self):
        start = time.time()
        nexttime = start + self.period
        while True:
            now = time.time()
            elapsed = now - start
            if elapsed > self.max_seconds:
                raise StopIteration
            else:
                yield self.pretty_time(elapsed)
            tosleep = nexttime - now
            if tosleep <= 0 or not self.delay:
                nexttime = now + self.period
            else:
                nexttime = now + tosleep + self.period
                time.sleep(tosleep)


def create_parser():
    """Command line parser"""
    parser = argparse.ArgumentParser()

    parser.add_argument("--problem", default="deephyper.benchmarks.b1.problem.Problem")
    parser.add_argument("--run", default="deephyper.benchmarks.b1.addition_rnn.run")

    parser.add_argument("--backend", default='tensorflow',
                        help="Keras backend module name"
                       )
    parser.add_argument('--max-evals', type=int, default=100,
                        help='maximum number of evaluations'
                       )

    parser.add_argument('--ga-individuals-per-worker', type=int, default=1,
                        help='Initial population is num_workers *'
                        ' ind-per-worker', dest='individuals_per_worker'
                       )
    parser.add_argument('--ga-num-gen', type=int, default=40)

    parser.add_argument('--learner', action='store',
                        dest='learner',
                        nargs='?', const=1, type=str, default='RF',
                        choices=["RF", "ET", "GBRT", "DUMMY", "GP"],
                        help='type of learner')

    parser.add_argument('--ambs-lie-strategy', action='store',
                        default="cl_max", choices=["cl_min", "cl_mean", "cl_max"])

    parser.add_argument('--ambs-acq-func', action='store',
                        default="gp_hedge", choices=["LCB", "EI", "PI","gp_hedge"])

    parser.add_argument('--from-checkpoint', default=None,
                        help='path of checkpoint file from a previous run'
                       )

    parser.add_argument('--eval-timeout-minutes', type=int, default=-1, help="Kill evals that take longer than this")
    parser.add_argument('--evaluator', default='local', help="'balsam' or 'local'")

    return parser

def load_attr_from(str_full_module):
    """
        Args:
            - str_full_module: (str) correspond to {module_name}.{attr}
        Return: the loaded attribut from a module.
    """
    split_full = str_full_module.split('.')
    str_module = '.'.join(split_full[:-1])
    str_attr = split_full[-1]
    module = import_module(str_module)
    return getattr(module, str_attr)

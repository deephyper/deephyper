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

class OptConfig:
    '''Optimizer-related options'''

    def __init__(self, args, problem=None):
        HERE = os.path.dirname(os.path.abspath(__file__)) # search dir
        package = os.path.basename(os.path.dirname(HERE)) # 'deephyper'

        self.backend = args.backend
        self.max_evals = args.max_evals
        self.individuals_per_worker = args.individuals_per_worker
        self.ga_num_gen = args.ga_num_gen
        self.evaluator = args.evaluator
        self.repeat_evals = args.repeat_evals
        self.num_workers = args.num_workers
        self.learner = args.learner
        self.amls_lie_strategy = args.amls_lie_strategy
        self.amls_acq_func = args.amls_acq_func

        self.model_path = args.model_path.strip()
        self.data_source = args.data_source.strip()
        self.stage_in_destination = args.stage_in_destination.strip()

        # for example, the default value of args.benchmark is "b1.addition_rnn"
        benchmark_directory = args.benchmark.split('.')[0] # "b1"
        self.benchmark = args.benchmark
        problem_module_name = f'{package}.benchmarks.{benchmark_directory}.problem'
        problem_module = import_module(problem_module_name)

        # get the path of the b1/addition_rnn.py file here:
        self.benchmark_module_name = f'{package}.benchmarks.{args.benchmark}'
        self.benchmark_filename = find_spec(self.benchmark_module_name).origin

        # create a problem instance and configure the skopt.Optimizer
        instance = problem_module.Problem()
        self.params = list(instance.params)
        self.starting_point = instance.starting_point

        spaceDict = instance.space
        self.space = [spaceDict[key] for key in self.params]

        # get the whole space dictionnary
        self.space_dict = instance.space

class OptConfigNas:
    '''Nas-related options'''

    def __init__(self, args, num_workers=None, problem=None):
        HERE = os.path.dirname(os.path.abspath(__file__)) # search dir
        package = os.path.basename(os.path.dirname(HERE)) # 'deephyper'

        self.backend = args.backend
        self.max_evals = args.max_evals
        self.individuals_per_worker = args.individuals_per_worker
        self.ga_num_gen = args.ga_num_gen
        self.evaluator = args.evaluator
        self.repeat_evals = args.repeat_evals
        self.learner = args.learner
        self.sync = args.sync

        self.model_path = args.model_path.strip()
        self.stage_in_destination = args.stage_in_destination.strip()

        # for example, the default value of args.benchmark is "mnistNast"
        self.benchmark = args.benchmark
        self.bench_package_name = f'{package}.benchmarks.{self.benchmark}'

        # load problem.py
        problem_module_name = f'{package}.benchmarks.{self.benchmark}.problem'
        problem_module = import_module(problem_module_name)

        # load load_data.py and the load_data function inside it
        load_data_module_name = f'{package}.benchmarks.{self.benchmark}.load_data'

        # run module which contain a run(param_dict) function which return 'something'
        self.run_module_name = args.run_module_name
        self.run_module = import_module(self.run_module_name) #run module

        # create a problem instance
        instance = problem_module.Problem()

        # get the whole space dictionnary
        self.config = instance.space
        self.config['load_data_module_name'] = load_data_module_name

def sk_optimizer_from_config(opt_config, random_state):
    from skopt import Optimizer
    #from deephyper.search.ExtremeGradientBoostingQuantileRegressor import \
         #ExtremeGradientBoostingQuantileRegressor
    from numpy import inf
    logger = logging.getLogger(__name__)
    kappa = 1.96

    if opt_config.learner in "RF ET GBRT XGB GP".split():
        n_init = opt_config.num_workers
    else:
        assert opt_config.learner == "DUMMY"
        n_init = inf

    if opt_config.learner in ["RF", "ET", "GBRT", "GP", "DUMMY"]:
        optimizer = Optimizer(
            opt_config.space,
            base_estimator=opt_config.learner,
            acq_optimizer='sampling',
            acq_func=opt_config.amls_acq_func,
            acq_func_kwargs={'kappa':kappa},
            random_state=random_state,
            n_initial_points=n_init
        )
    elif opt_config.learner == "XGB":
        optimizer = Optimizer(
            opt_config.space,
            base_estimator=ExtremeGradientBoostingQuantileRegressor(),
            acq_optimizer='sampling',
            acq_func='gp_hedge',
            acq_func_kwargs={'kappa':kappa},
            random_state=random_state,
            n_initial_points=n_init
        )
    else:
        raise ValueError(f"Unknown learner type {opt_config.learner}")
    logger.info("Creating skopt.Optimizer with %s base_estimator" % opt_config.learner)
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
        '''Format time string'''
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
    '''Command line parser'''
    parser = argparse.ArgumentParser()

    parser.add_argument("--benchmark", default='b1.addition_rnn',
                        help="if hyperparameters search : name of benchmark module (e.g. b1.addition_rnn \n if neural architecture search : name of benchmark package (e.g. mnistNas))"
                       )
    parser.add_argument("--backend", default='tensorflow',
                        help="Keras backend module name"
                       )
    parser.add_argument('--max-evals', type=int, default=100,
                        help='maximum number of evaluations'
                       )
    parser.add_argument('--run', help='optional tag appended to run files')
    # parser.add_argument('--start_num_layers', type=int, default=2,
    #                     help='Number of layers to start for initially'
    #                     )
    # parser.add_argument('--max_num_layers', type=int, default=2,
    #                     help='Max number of layers to try'
    #                     )
    parser.add_argument('--ga-individuals-per-worker', type=int, default=1,
                        help='Initial population is num_workers *'
                        ' ind-per-worker', dest='individuals_per_worker'
                       )
    parser.add_argument('--ga-num-gen', type=int, default=40)

    parser.add_argument('--learner', action='store',
                        dest='learner',
                        nargs='?', const=1, type=str, default='XGB',
                        choices=["XGB", "RF", "ET", "GBRT", "DUMMY", "GP"],
                        help='type of learner')

    parser.add_argument('--amls-lie-strategy', action='store',
                        default="cl_max", choices=["cl_min", "cl_mean", "cl_max"])

    parser.add_argument('--amls-acq-func', action='store',
                        default="gp_hedge", choices=["LCB", "EI", "PI","gp_hedge"])

    parser.add_argument('--from-checkpoint', default=None,
                        help='path of checkpoint file from a previous run'
                       )
    parser.add_argument('--evaluator', default='balsam')
    parser.add_argument('--repeat-evals', action='store_true',
                        help='Re-evaluate points visited by hyperparameter optimizer'
                       )
    parser.add_argument('--model_path', help="path from which models are loaded/saved", default='savepoint/model')
    parser.add_argument('--data_source', help="location of dataset to load", default='')
    parser.add_argument('--stage_in_destination', help="if provided; cache data at this location",
                        default='')
    parser.add_argument('--eval-timeout-minutes', type=int, default=-1, help="Kill evals that take longer than this")

    # Args for nas
    parser.add_argument('--run_module_name', default='search.nas', help="name\
        of run module (e.g. mnistNas.run1)")
    parser.add_argument('--sync', dest='sync', action='store_true', default=False)

    return parser

def load_attr_from(str_full_module):
    '''
        Args:
            - str_full_module: (str) correspond to {module_name}.{attr}
        Return: the loaded attribut from a module.
    '''
    split_full = str_full_module.split('.')
    str_module = '.'.join(split_full[:-1])
    str_attr = split_full[-1]
    module = import_module(str_module)
    return getattr(module, str_attr)

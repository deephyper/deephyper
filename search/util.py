import argparse
import os
import csv
import time
import logging
from importlib import import_module
from importlib.util import find_spec

class OptConfig:
    '''Optimizer-related options'''

    def __init__(self, args):
        HERE = os.path.dirname(os.path.abspath(__file__)) # search dir
        package = os.path.basename(os.path.dirname(HERE)) # 'deephyper'

        self.backend = args.backend
        self.max_evals = args.max_evals 
        self.evaluator = args.evaluator
        self.repeat_evals = args.repeat_evals
        self.num_workers = args.num_workers
        
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

def conf_logger():
    logger = logging.getLogger('deephyper')

    handler = logging.FileHandler('deephyper.log')
    formatter = logging.Formatter(
        '%(asctime)s|%(process)d|%(levelname)s|%(name)s:%(lineno)s] %(message)s', 
        "%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.info("\n\nLoading Deephyper\n--------------")
    return logger

def elapsed_timer(max_runtime_minutes=None, service_period=2):
    '''Iterator over elapsed seconds; ensure delay of service_period
    Raises StopIteration when time is up'''
    if max_runtime_minutes is None:
        max_runtime_minutes = float('inf')
        
    max_runtime = max_runtime_minutes * 60.0

    start = time.time()
    nexttime = start + service_period
    while True:
        now = time.time()
        elapsed = now - start
        if elapsed > max_runtime+0.5:
            raise StopIteration
        else:
            yield elapsed
        tosleep = nexttime - now
        if tosleep <= 0:
            nexttime = now + service_period
        else:
            nexttime = now + tosleep + service_period
            time.sleep(tosleep)

def pretty_time(seconds):
    '''Format time string'''
    seconds = round(seconds)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return "%02d:%02d:%02d" % (hours,minutes,seconds)

def create_parser():
    '''Command line parser'''
    parser = argparse.ArgumentParser()

    parser.add_argument("--benchmark", default='b1.addition_rnn',
                        help="name of benchmark module (e.g. b1.addition_rnn)"
                       )
    parser.add_argument("--backend", default='tensorflow',
                        help="Keras backend module name"
                       )
    parser.add_argument('--max-evals', type=int, default=100,
                        help='maximum number of evaluations'
                       )
    parser.add_argument('--num-workers', type=int, default=10,
                        help='Number of points to ask for initially'
                       )
    parser.add_argument('--from-checkpoint', default=None,
                        help='path of checkpoint file from a previous run'
                       )
    parser.add_argument('--evaluator', default='balsam')
    parser.add_argument('--repeat-evals', action='store_true',
                        help='Re-evaluate points visited by hyperparameter optimizer'
                       )
    return parser

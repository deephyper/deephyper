from importlib import import_module
from importlib.util import find_spec
import os

from skopt import Optimizer
from deephyper.search.ExtremeGradientBoostingQuantileRegressor import ExtremeGradientBoostingQuantileRegressor

class OptConfig:
    '''Optimizer-related options'''

    SEED = 12345

    def __init__(self, args,
                 base_estimator=ExtremeGradientBoostingQuantileRegressor(),
                 acq_optimizer='sampling',
                 acq_func='LCB',
                 acq_func_kwargs={'kappa':0},
                 random_state=SEED
                 ):

        here = os.path.dirname(os.path.abspath(__file__)) # search dir
        package = os.path.basename(os.path.dirname(here)) # 'deephyper'

        self.backend = args.backend
        self.max_evals = args.max_evals 
        self.repeat_evals = args.repeat_evals
        
        # THIS IS WHERE THE BENCHMARK IS AUTO-LOCATED
        # for example, the default value of args.benchmark is "b1.addition_rnn"
        # ----------------------------------------------------------------------
        benchmark_directory = args.benchmark.split('.')[0] # "b1"
        problem_module_name = f'{package}.benchmarks.{benchmark_directory}.problem'
        problem_module = import_module(problem_module_name)

        # get the path of the b1/addition_rnn.py file here:
        benchmark_module_name = f'{package}.benchmarks.{args.benchmark}'
        self.benchmark_filename = find_spec(benchmark_module_name).origin
        self.benchmark_module = import_module(benchmark_module_name)
        
        # create a problem instance and configure the skopt.Optimizer
        instance = problem_module.Problem()
        self.params = list(instance.params)
        self.starting_point = instance.starting_point
        
        spaceDict = instance.space
        space = [spaceDict[key] for key in cfg.params]
        
        self.optimizer = Optimizer(space, base_estimator=base_estimator,
                                   acq_optimizer=acq_optimizer,
                                   acq_func=acq_func,
                                   acq_func_kwargs=acq_func_kwargs,
                                   random_state=SEED
                                  )
        return cfg

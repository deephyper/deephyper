from importlib import import_module
from importlib.util import find_spec

import deap
import deap.gp
import deap.benchmarks
from deap import base
from deap import creator
from deap import tools
from deap import algorithms

from skopt import Optimizer
from deephyper.search.ExtremeGradientBoostingQuantileRegressor import ExtremeGradientBoostingQuantileRegressor

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import numpy as np
import random

SEED = 12345                # Optimizer initialized with this random seed

def uniform(lower_list, upper_list, dimensions):
    """Fill array """
    if hasattr(lower_list, '__iter__'):
        return [random.uniform(lower, upper) 
                for lower, upper in zip(lower_list, upper_list)]
    else:
        return [random.uniform(lower_list, upper_list) 
                for _ in range(dimensions)]

class OptConfig:
    '''Optimizer-related options'''

    def __init__(self, args):
        package = os.path.basename(os.path.dirname(HERE)) # 'deephyper'

        self.backend = args.backend
        self.max_evals = args.max_evals 
        self.repeat_evals = args.repeat_evals
        self.num_workers = args.num_workers
        
        # for example, the default value of args.benchmark is "b1.addition_rnn"
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
        self.space = [spaceDict[key] for key in cfg.params]

class SamplingOptimizer:
    def __init__(self, opt_config, evaluator,
                 base_estimator=ExtremeGradientBoostingQuantileRegressor(),
                 acq_optimizer='sampling',
                 acq_func='LCB',
                 acq_func_kwargs={'kappa':0},
                 random_state=SEED):

        self.starting_point = opt_config.starting_point
        self.max_evals = opt_config.max_evals
        self.repeat_evals = opt_config.repeat_evals

        self.evaluator = evaluator
        self.optimizer = Optimizer(opt_config.space, base_estimator=base_estimator,
                                   acq_optimizer=acq_optimizer,
                                   acq_func=acq_func,
                                   acq_func_kwargs=acq_func_kwargs,
                                   random_state=random_state)

    def next_points(self, eval_counter, my_jobs):
        '''Query optimizer for the next set of points to evaluate'''
        if self.evaluator.counter >= self.max_evals:
            logger.debug("Reached max_evals; no longer starting new runs")
            return []

        if self.starting_point is not None:
            XX = [self.starting_point]
            self.starting_point = None
            additional_pts = self.optimizer.ask(n_points=evaluator.num_workers-1)
            XX.extend(additional_pts)
        elif self.evaluator.num_free_workers() > 0:
            XX = self.optimizer.ask(n_points=1)
        else:
            XX = []

        if not self.repeat_evals:
            XX = [x for x in XX if json.dumps(x, cls=Encoder) not in
                  self.evaluator.evals()]
        return XX

class GAOptimizer:
    def __init__(self, opt_config, evaluator,
                 seed=SEED,
                 CXPB=0.5,
                 MUTPB=0.2,
                 NGEN=40):
        self.space = opt_config.space
        self.space_encoder = SpaceEncoder(self.space)
        self.evaluator = evaluator

        self.max_evals = opt_config.max_evals
        self.CXPB = CXPB
        self.MUTPB = MUTPB
        self.NGEN = NGEN

        self.IND_SIZE = len(self.space)
        LOWER = [0.0] * self.IND_SIZE
        UPPER = [1.0] * self.IND_SIZE

        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        self.toolbox = base.Toolbox()

        random.seed(seed)

        self.toolbox.register("uniformparams", uniform, LOWER, UPPER, IND_SIZE)
        self.toolbox.register("Individual",tools.initIterate,
                              creator.Individual,toolbox.uniformparams)
        self.toolbox.register("population", tools.initRepeat, list, toolbox.Individual)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
        self.toolbox.register("select", tools.selTournament, tournsize=3)


class SpaceEncoder:
    def __init__(self, space):
        self.space = space
        self.encoders = []
        self.ttypes = []
        self.encode_space() 

    def encode_space(self):
        for p in self.space:
            enc, ttype = self.encode(p)
            self.encoders.append(enc)
            self.ttypes.append(ttype)

    def encode(self, val):
        ttype = 'i'
        if isinstance(val, list):
            encoder = LabelEncoder()
            encoder.fit(val)
            ttype = 'c'
        else:
            encoder = MinMaxScaler()
            encoder.fit(np.asarray(val).reshape(-1, 1))
            if isinstance(val[0], float):
                ttype = 'f'
        return encoder, ttype

    def decode_point(self, point):
        result = [ self.decode(point[i], self.encoders[i]) for i in range(len(point)) ]
        for i in range(len(point)): 
            if self.ttypes[i] == 'i':
                result[i] = int(round(result[i])) 
        return result

    def decode(self, enc_val, encoder):
        dec_val = enc_val
        if hasattr(encoder, 'classes_'):
            bins = np.linspace(0.0, 1.0, num=1+len(list(encoder.classes_)))
            dec_val = max(0, np.digitize(enc_val, bins, right=True) - 1)
        dec_val = np.asscalar(encoder.inverse_transform(dec_val))
        return dec_val

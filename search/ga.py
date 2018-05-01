from copy import copy
import logging
import pickle
import os
import signal
import sys

import deap
import deap.gp
import deap.benchmarks
from deap import base
from deap import creator
from deap import tools
from deap import algorithms

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import numpy as np
import random


HERE = os.path.dirname(os.path.abspath(__file__)) # search dir
top  = os.path.dirname(os.path.dirname(HERE)) # directory containing deephyper
sys.path.append(top)

from deephyper.evaluators import evaluate
from deephyper.search import util

logger = util.conf_logger('deephyper.search.ga')

SEED = 12345
CHECKPOINT_INTERVAL = 1    # How many generations between optimizer checkpoints
SERVICE_PERIOD = 2
    
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

def uniform(lower_list, upper_list, dimensions):
    """Fill array """
    if hasattr(lower_list, '__iter__'):
        return [random.uniform(lower, upper) 
                for lower, upper in zip(lower_list, upper_list)]
    else:
        return [random.uniform(lower_list, upper_list) 
                for _ in range(dimensions)]


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
        try:
            result = [ self.decode(point[i], self.encoders[i]) for i in range(len(point)) ]
        except ValueError:
            print("GOT VALUE ERROR WHEN TRYING TO DECODE", point)
            raise
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


class GAOptimizer:

    def __init__(self, opt_config, seed=SEED, CXPB=0.5, MUTPB=0.2):
        self.SEED = seed
        self.CXPB = CXPB
        self.MUTPB = MUTPB
        self.NGEN = opt_config.ga_num_gen

        pop_size = opt_config.num_workers * opt_config.individuals_per_worker
        self.INIT_POP_SIZE = pop_size
        self.IND_SIZE = len(opt_config.space)

        self.toolbox = None
        self.space_encoder = SpaceEncoder(opt_config.space)

        self._setup()

        self.current_gen = 0
        self.pop = None
        self.halloffame = tools.HallOfFame(maxsize=1)
        self.logbook = tools.Logbook()

    def _check_bounds(self, min, max):
        def decorator(func):
            def wrapper(*args, **kargs):
                offspring = func(*args, **kargs)
                for child in offspring:
                    for i in range(len(child)):
                        if child[i] > max:
                            child[i] = max
                        elif child[i] < min:
                            child[i] = min
                return offspring
            return wrapper
        return decorator

    def _setup(self):
        random.seed(self.SEED)

        self.toolbox = base.Toolbox()
        
        LOWER = [0.0] * self.IND_SIZE
        UPPER = [1.0] * self.IND_SIZE
        self.toolbox.register("uniformparams", uniform, LOWER, UPPER, self.IND_SIZE)
        self.toolbox.register("Individual",tools.initIterate,
                              creator.Individual, self.toolbox.uniformparams)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.Individual)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.decorate("mate", self._check_bounds(0.0, 1.0))
        self.toolbox.decorate("mutate", self._check_bounds(0.0, 1.0))

        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("avg", np.mean)
        self.stats.register("min", np.min)

    def record_generation(self, num_evals):
        self.halloffame.update(self.pop)
        record = self.stats.compile(self.pop)
        self.logbook.record(gen=self.current_gen, evals=num_evals, **record)

    def __getstate__(self):
        d = copy(self.__dict__)
        d['toolbox'] = None
        d['stats'] = None
        return d
    
    def __setstate__(self, d):
        self.__dict__ = d
        self._setup()


def evaluate_fitnesses(individuals, opt, evaluator, timeout_minutes):
    points = list(map(opt.space_encoder.decode_point, individuals))
    for x in points: evaluator.add_eval(x)
    logger.info(f"Waiting on {len(points)} individual fitness evaluations")
    results = evaluator.await_evals(points, timeout_sec=timeout_minutes*60)

    for ind, (x,fit) in zip(individuals, results):
        ind.fitness.values = (fit,)


def save_checkpoint(opt_config, optimizer, evaluator):
    if evaluator.counter == 0: return
    data = {}
    data['opt_config'] = opt_config
    data['optimizer'] = optimizer
    data['evaluator'] = evaluator
    
    fname = f'{opt_config.benchmark}.pkl'
    with open(fname, 'wb') as fp: pickle.dump(data, fp)

    evaluator.dump_evals()
    logger.info(f"Checkpointed run in {os.path.abspath(fname)}")


def load_checkpoint(chk_path):
    chk_path = os.path.abspath(os.path.expanduser(chk_path))
    assert os.path.exists(chk_path), "No such checkpoint file"
    with open(chk_path, 'rb') as fp: data = pickle.load(fp)
    
    cfg, opt, evaluator = data['opt_config'], data['optimizer'], data['evaluator']

    cfg.num_workers = args.num_workers
    logger.info(f"Resuming GA from checkpoint in {chk_path}")
    logger.info(f"On eval {evaluator.counter}")
    return cfg, opt, evaluator


def main(args):
    if args.from_checkpoint:
        chk_path = args.from_checkpoint
        cfg, opt, evaluator = load_checkpoint(chk_path)
    else:
        cfg = util.OptConfig(args)
        opt = GAOptimizer(cfg)
        evaluator = evaluate.create_evaluator(cfg)
        logger.info(f"Starting new run with {cfg.benchmark_module_name}")


    timer = util.DelayTimer(max_minutes=None, period=SERVICE_PERIOD)
    timer = iter(timer)
    elapsed_str = next(timer)
    chkpoint_counter = 0

    logger.info("Hyperopt GA driver starting")
    logger.info(f"Elapsed time: {elapsed_str}")
    
    # Gracefully handle shutdown
    def handler(signum, stack):
        evaluator.stop()
        logger.info('Received SIGINT/SIGTERM')
        save_checkpoint(cfg, opt, evaluator)
        sys.exit(0)

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)

    if opt.pop is None:
        logger.info("Generating initial population")
        logger.info(f"{opt.INIT_POP_SIZE} individuals")
        opt.pop = opt.toolbox.population(n=opt.INIT_POP_SIZE)
        individuals = opt.pop
        evaluate_fitnesses(individuals, opt, evaluator, args.eval_timeout_minutes)
        opt.record_generation(num_evals=len(opt.pop))
        
        with open('ga_logbook.log', 'w') as fp:
            fp.write(str(opt.logbook))
        print("best:", opt.halloffame[0])
    
    while opt.current_gen < opt.NGEN:
        opt.current_gen += 1
        time_str = next(timer)
        logger.info(f"Generation {opt.current_gen} out of {opt.NGEN}")
        logger.info(f"Elapsed time: {elapsed_str}")

        # Select the next generation individuals
        offspring = opt.toolbox.select(opt.pop, len(opt.pop))
        # Clone the selected individuals
        offspring = list(map(opt.toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < opt.CXPB:
                opt.toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < opt.MUTPB:
                opt.toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        logger.info(f"Evaluating {len(invalid_ind)} invalid individuals")
        evaluate_fitnesses(invalid_ind, opt, evaluator, args.eval_timeout_minutes)
        
        # The population is entirely replaced by the offspring
        opt.pop[:] = offspring

        opt.record_generation(num_evals=len(invalid_ind))

        with open('ga_logbook.log', 'w') as fp:
            fp.write(str(opt.logbook))
        print("best:", opt.halloffame[0])
        

        chkpoint_counter += 1
        if chkpoint_counter >= CHECKPOINT_INTERVAL:
            save_checkpoint(cfg, opt, evaluator)
            chkpoint_counter = 0
        sys.stdout.flush()
    
    # EXIT
    logger.info('Hyperopt GA driver finishing')
    save_checkpoint(cfg, opt, evaluator)


if __name__ == "__main__":
    parser = util.create_parser()
    args = parser.parse_args()
    main(args)

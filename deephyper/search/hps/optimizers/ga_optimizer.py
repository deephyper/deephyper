import random
import numpy as np
from copy import copy

from deap import base, creator, tools
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

SEED = 12345

class GAOptimizer:

    def __init__(self, problem, num_workers, args, seed=SEED, CXPB=0.5, MUTPB=0.2):
        self.SEED = seed
        self.CXPB = CXPB
        self.MUTPB = MUTPB
        self.NGEN = args.ga_num_gen

        pop_size = num_workers * args.individuals_per_worker
        self.INIT_POP_SIZE = pop_size
        self.IND_SIZE = len(problem.space)

        self.toolbox = None
        self.space_encoder = SpaceEncoder(problem.space.values())

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
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

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
        dec_val = encoder.inverse_transform(np.array([dec_val]).reshape(1, -1))
        return np.asscalar(dec_val)

def uniform(lower_list, upper_list, dimensions):
    """Fill array """
    if hasattr(lower_list, '__iter__'):
        return [random.uniform(lower, upper)
                for lower, upper in zip(lower_list, upper_list)]
    else:
        return [random.uniform(lower_list, upper_list)
                for _ in range(dimensions)]

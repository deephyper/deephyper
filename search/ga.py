#!/usr/bin/env python
from __future__ import print_function
from mpi4py import MPI
import random
import numpy

import deap
import deap.gp
import deap.benchmarks
from deap import base
from deap import creator
from deap import tools
from deap import algorithms
import re
import os
import sys
import time
import json
import math
from skopt import Optimizer
from utils import *
import os
import argparse
seed = 12345


from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.pipeline import Pipeline

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

def create_parser():
    'command line parser for keras'

    parser = argparse.ArgumentParser(add_help=True)
    group = parser.add_argument_group('required arguments')
    parser.add_argument('-v', '--version', action='version',
                        version='%(prog)s 0.1')
    parser.add_argument("--prob_dir", nargs='?', type=str,
                        default='../problems/prob1',
                        help="problem directory")
    parser.add_argument("--exp_dir", nargs='?', type=str,
                        default='../experiments',
                        help="experiments directory")
    parser.add_argument("--exp_id", nargs='?', type=str,
                        default='exp-01',
                        help="experiments id")
    parser.add_argument('--max_evals', action='store', dest='max_evals',
                        nargs='?', const=2, type=int, default='30',
                        help='maximum number of evaluations')
    parser.add_argument('--max_time', action='store', dest='max_time',
                        nargs='?', const=1, type=float, default='60',
                        help='maximum time in secs')

    return(parser)

parser = create_parser()
cmdline_args = parser.parse_args()
param_dict = vars(cmdline_args)

prob_dir = param_dict['prob_dir'] #'/Users/pbalapra/Projects/repos/2017/dl-hps/benchmarks/test'
exp_dir = param_dict['exp_dir'] #'/Users/pbalapra/Projects/repos/2017/dl-hps/experiments'
eid = param_dict['exp_id'] #'exp-01'
max_evals = param_dict['max_evals']
max_time = param_dict['max_time']


exp_dir = exp_dir+'/'+eid
jobs_dir = exp_dir+'/jobs'
results_dir = exp_dir+'/results'
results_json_fname = exp_dir+'/'+eid+'_results.json'
results_csv_fname = exp_dir+'/'+eid+'_results.csv'

sys.path.insert(0, prob_dir)
import problem as problem
from evaluate import evaluate

instance = problem.Problem()
spaceDict = instance.space
params = instance.params
starting_point = instance.starting_point

space = [spaceDict[key] for key in params]



random.seed(seed)
POP_SIZE = 10
OFFSPRING_SIZE = 10

NGEN = max_evals/POP_SIZE
ALPHA = POP_SIZE
MU = OFFSPRING_SIZE
LAMBDA = OFFSPRING_SIZE
CXPB = 0.7
MUTPB = 0.3
ETA = 10.0


space_enc = SpaceEncoder(space)

IND_SIZE = len(space)
LOWER = [0.0] * IND_SIZE 
UPPER = [1.0] * IND_SIZE

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

def uniform(lower_list, upper_list, dimensions):
    """Fill array """

    if hasattr(lower_list, '__iter__'):
        return [random.uniform(lower, upper) for lower, upper in
                zip(lower_list, upper_list)]
    else:
        return [random.uniform(lower_list, upper_list)
                for _ in range(dimensions)]

toolbox = base.Toolbox()
toolbox.register("uniformparams", uniform, LOWER, UPPER, IND_SIZE)
toolbox.register("Individual",tools.initIterate,creator.Individual,toolbox.uniformparams)
toolbox.register("population", tools.initRepeat, list, toolbox.Individual)

def evaluate_ga(x, eval_counter):
    result = evaluate(x, eval_counter, params, prob_dir, jobs_dir, results_dir)
    y = result['cost']
    return y,

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate_ga", evaluate_ga)

def main():
    resultsList = []
    eval_counter = 0
    start_time = time.time()
    pop = toolbox.population(n=50)
    CXPB, MUTPB, NGEN = 0.5, 0.2, 40
    for x in pop:
        task = {}
        task['x'] = space_enc.decode_point(x)
        print(task['x'])
    # Evaluate the entire population
    #fitnesses = map(toolbox.evaluate_ga, pop)
    fitnesses = []
    for x in pop:
        task = {}
        task['x'] = space_enc.decode_point(x)
        task['eval_counter'] = eval_counter
        task['start_time'] = float(time.time() - start_time)
        fitness = toolbox.evaluate_ga(task['x'], task['eval_counter'])
        task['end_time'] = float(time.time() - start_time)
        eval_counter = eval_counter + 1
        fitnesses.append(fitness)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    for g in range(NGEN):
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = map(toolbox.clone, offspring)

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = []
        for x in invalid_ind:
            task = {}
            task['x'] = space_enc.decode_point(x)
            task['eval_counter'] = eval_counter
            task['start_time'] = float(time.time() - start_time)
            fitness = toolbox.evaluate_ga(task['x'], task['eval_counter'])
            task['end_time'] = float(time.time() - start_time)
            eval_counter = eval_counter + 1
            fitnesses.append(fitness)
            resultsList.append(task)

        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
            print(ind.fitness.values)

        # The population is entirely replaced by the offspring
        pop[:] = offspring
    saveResults(resultsList, results_json_fname, results_csv_fname)
    return pop

if __name__ == '__main__':
    main()


#!/usr/bin/env python
"""Demonstrate the task-pull paradigm for high-throughput computing
using mpi4py. Task pull is an efficient way to perform a large number of
independent tasks when there are more tasks than processors, especially
when the run times vary for each task.

This code is over-commented for instructional purposes.

This example was contributed by Craig Finch (cfinch@ieee.org).
Inspired by http://math.acadiau.ca/ACMMaC/Rmpi/index.html
"""
from __future__ import print_function

from mpi4py import MPI
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
instance = problem.Problem()
spaceDict = instance.space
params = instance.params
starting_point = instance.starting_point

def enum(*sequential, **named):
    """Handy way to fake an enumerated type in Python
    http://stackoverflow.com/questions/36932/how-can-i-represent-an-enum-in-python
    """
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type('Enum', (), enums)

# Define MPI message tags
tags = enum('READY', 'DONE', 'EXIT', 'START')

# Initializations and preliminaries
comm = MPI.COMM_WORLD   # get MPI communicator object
size = comm.size        # total number of processes
rank = comm.rank        # rank of this process
status = MPI.Status()   # get MPI status object

# Master process executes code below
if rank == 0:
    start_time = time.time()
    for dir_name in [exp_dir, jobs_dir, results_dir]: 
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

    num_workers = size - 1
    closed_workers = 0
    space = [spaceDict[key] for key in params]
    eval_counter = 0


    parDict = {}
    evalDict = {}
    resultsList = []
    parDict['kappa'] = 0
    opt = Optimizer(space, base_estimator=RF, acq_optimizer='sampling',
                    acq_func='LCB', acq_func_kwargs=parDict, random_state=seed)
    print("Master starting with %d workers" % num_workers)
    while closed_workers < num_workers:
        data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        source = status.Get_source()
        tag = status.Get_tag()
        elapsed_time = float(time.time() - start_time)
        print('elapsed_time:%1.3f'%elapsed_time)
        if tag == tags.READY:
            if eval_counter < max_evals and elapsed_time < max_time:
                # Worker is ready, so send it a task
                if starting_point is not None:
                    x = starting_point
                    starting_point = None
                else:
                    x = opt.ask(n_points=1)[0]
                key = str(x)
                print('sample %s' % key)
                if key in evalDict.keys():
                    print('%s already evalauted' % key)
                evalDict[key] = None
                task = {}
                task['x'] = x
                task['eval_counter'] = eval_counter
                task['start_time'] = elapsed_time
                print("Sending task %d to worker %d" % (eval_counter, source))
                comm.send(task, dest=source, tag=tags.START)
                eval_counter = eval_counter + 1
            else:
                comm.send(None, dest=source, tag=tags.EXIT)
        elif tag == tags.DONE:
            result = data
            result['end_time'] = elapsed_time
            print("Got data from worker %d" % source)
            print(result)
            resultsList.append(result)
            x = result['x']
            y = result['cost']
            opt.tell(x, y)
        elif tag == tags.EXIT:
            print("Worker %d exited." % source)
            closed_workers = closed_workers + 1
    print('Search finishing')
    # print(resultsList)
    # print(json.dumps(resultsList, indent=4, sort_keys=True))
    
    saveResults(resultsList, results_json_fname, results_csv_fname)

else:
    # Worker processes execute code below
    name = MPI.Get_processor_name()
    print("worker with rank %d on %s." % (rank, name))
    while True:
        comm.send(None, dest=0, tag=tags.READY)
        task = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        tag = status.Get_tag()
        if tag == tags.START:
            print(task)
            result = evaluatePoint(task['x'], task['eval_counter'], params, prob_dir, jobs_dir, results_dir)
            result['start_time'] = task['start_time']
            comm.send(result, dest=0, tag=tags.DONE)
        elif tag == tags.EXIT:
            break
    comm.send(None, dest=0, tag=tags.EXIT)

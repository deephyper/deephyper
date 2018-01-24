import argparse
import csv
import json
from numpy import float64, int64
import os
from pprint import pprint
import signal
from importlib import import_module
from importlib.util import find_spec
import sys
import time

from skopt import Optimizer

import balsam.launcher.dag as dag
from balsam.service.models import BalsamJob, END_STATES

from dl_hps.search.ExtremeGradientBoostingQuantileRegressor import ExtremeGradientBoostingQuantileRegressor

SEED = 12345
MAX_QUEUED_TASKS = 128
SERVICE_PERIOD = 2

def elapsed_timer(max_runtime=None):
    '''Iterator over elapsed seconds; ensure delay of SERVICE_PERIOD
    Raises StopIteration when time is up'''
    if max_runtime is None: 
        max_runtime = float('inf')

    start = time.time()
    nexttime = start + SERVICE_PERIOD
    while True:
        now = time.time()
        elapsed = now - start
        if elapsed > max_runtime+0.5:
            raise StopIteration
        else:
            yield elapsed
        tosleep = nexttime - now
        if tosleep <= 0:
            nexttime = now + SERVICE_PERIOD
        else:
            nexttime = now + tosleep + SERVICE_PERIOD
            time.sleep(tosleep)


def pretty_time(seconds):
    '''Format time string'''
    seconds = round(seconds)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return "%02d:%02d:%02d" % (hours,minutes,seconds)


def create_parser():
    '''Command line parser for Keras'''
    parser = argparse.ArgumentParser(add_help=True)
    group = parser.add_argument_group('required arguments')
    parser.add_argument('-v', '--version', action='version',
                        version='%(prog)s 0.1')
    parser.add_argument("--benchmark", nargs='?', type=str,
                        default='b1.addition_rnn',
                        help="name of benchmark module (e.g. b1.addition_rnn)")
    parser.add_argument("--backend", nargs='?', type=str,
                        default='tensorflow',
                        help="Keras backend module name")
    parser.add_argument("--exp_dir", nargs='?', type=str,
                        default='../experiments',
                        help="experiments directory")
    parser.add_argument("--exp_id", nargs='?', type=str,
                        default='exp-01',
                        help="experiments id")
    parser.add_argument('--max_evals', action='store', dest='max_evals',
                        nargs='?', const=2, type=int, default='10',
                        help='maximum number of evaluations')
    parser.add_argument('--max_time', action='store', dest='max_time',
                        nargs='?', const=1, type=float, default='60',
                        help='maximum time in secs')
    return parser


def configureOptimizer(args):
    '''Return a Config object with skopt.Optimizer and various options configured'''
    class Config: pass
    P = Config()

    top = args.benchmark.split('.')[0]
    problem = import_module(f'dl_hps.benchmarks.{top}.problem')
    P.benchmark = find_spec(f'dl_hps.benchmarks.{args.benchmark}').origin # b1.addition_rnn
    P.backend = args.backend

    P.exp_dir = os.path.expanduser(args.exp_dir) #'/Users/pbalapra/Projects/repos/2017/dl-hps/experiments'
    P.eid = args.exp_id  #'exp-01'
    P.max_evals = args.max_evals 
    P.max_time = args.max_time

    P.exp_dir = os.path.join(P.exp_dir, str(P.eid))
    if not os.path.exists(P.exp_dir):
        os.makedirs(P.exp_dir)

    P.results_json_fname = os.path.join(P.exp_dir, f"{P.eid}_results.json")
    P.results_csv_fname = os.path.join(P.exp_dir, f"{P.eid}_results.csv")
    
    instance = problem.Problem()
    P.params = list(instance.params)
    P.starting_point = instance.starting_point
    
    spaceDict = instance.space
    space = [spaceDict[key] for key in P.params]
    
    parDict = {}
    parDict['kappa'] = 0
    P.optimizer = Optimizer(space, base_estimator=ExtremeGradientBoostingQuantileRegressor(), acq_optimizer='sampling',
                    acq_func='LCB', acq_func_kwargs=parDict, random_state=SEED)
    return P


def create_job(x, eval_counter, cfg):
    '''Add a new evaluatePoint job to the Balsam DB'''
    task = {}
    task['x'] = x
    task['params'] = cfg.params
    task['benchmark'] = cfg.benchmark
    task['backend'] = cfg.backend

    for i, val in enumerate(x):
        if type(val) is int64: x[i]   = int(val)
        if type(val) is float64: x[i] = float(val)

    print(f"Adding task {eval_counter} to job DB")
    jname = f"task{eval_counter}"
    fname = f"{jname}.dat"

    with open(fname, 'w') as fp:
        fp.write(json.dumps(task))

    child = dag.spawn_child(name=jname, workflow="dl-hps",
                application="eval_point", wall_time_minutes=2,
                num_nodes=1, ranks_per_node=1,
                input_files=f"{jname}.dat", 
                application_args=f"{jname}.dat",
                wait_for_parents=False
               )
    return child.job_id

def saveResults(resultsList, json_fname, csv_fname):
    print(resultsList)
    print(json.dumps(resultsList, indent=4, sort_keys=True))
    with open(json_fname, 'w') as outfile:
        json.dump(resultsList, outfile, indent=4, sort_keys=True)

    keys = resultsList[0].keys()
    with open(csv_fname, 'w') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(resultsList)

def main():
    '''Service loop: add jobs; read results; drive optimizer'''
    parser = create_parser()
    args = parser.parse_args()
    cfg = configureOptimizer(args)
    opt = cfg.optimizer

    timer = elapsed_timer(max_runtime=cfg.max_time)
    eval_counter = 0

    evalDict = {}
    resultsList = []
    my_jobs = []
    finished_jobs = []

    # Gracefully handle shutdown
    SIG_TERMINATE = False
    def handler(signum, stack):
        print('Received SIGINT/SIGTERM')
        SIG_TERMINATE = True
        saveResults(resultsList, cfg.results_json_fname, cfg.results_csv_fname)
        sys.exit(0)

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)

    print("Hyperopt driver starting")

    for elapsed_seconds in timer:
        print("Elapsed time:", pretty_time(elapsed_seconds))
        if len(finished_jobs) == cfg.max_evals: break
        
        # Read in new results
        new_jobs = BalsamJob.objects.filter(job_id__in=my_jobs)
        new_jobs = new_jobs.filter(state="JOB_FINISHED")
        new_jobs = new_jobs.exclude(job_id__in=finished_jobs)
        for job in new_jobs:
            result = json.loads(job.read_file_in_workdir('result.dat'))
            result['run_time'] = job.runtime_seconds
            print(f"Got data from {job.cute_id}")
            pprint(result)
            resultsList.append(result)
            finished_jobs.append(job.job_id)
            x = result['x']
            y = result['cost']
            opt.tell(x, y)
        
        # Which points, and how many, are next?
        if cfg.starting_point is not None:
            XX = [cfg.starting_point]
            cfg.starting_point = None
        elif eval_counter < cfg.max_evals:
            already_active = BalsamJob.objects.filter(job_id__in=my_jobs)
            already_active = already_active.exclude(state__in=END_STATES).count()
            num_tocreate = max(MAX_QUEUED_TASKS - already_active, 0)
            num_tocreate = min(num_tocreate, cfg.max_evals - eval_counter)
            XX = opt.ask(n_points=num_tocreate) if num_tocreate else []
        else:
            XX = []
                
        # Create a BalsamJob for each point
        for x in XX:
            eval_counter += 1
            key = str(x)
            if key in evalDict: print(f"{key} already submitted!")
            evalDict[key] = None
            jobid = create_job(x, eval_counter, cfg)
            my_jobs.append(jobid)
    
    print('Hyperopt driver finishing')
    saveResults(resultsList, cfg.results_json_fname, cfg.results_csv_fname)

if __name__ == "__main__":
    main()

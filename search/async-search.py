import argparse
import csv
import json
from math import isnan
import os
from pprint import pprint
import pickle
from re import findall
import signal
import sys
import time

from deephyper.evaluate import Encoder, LocalEvaluator


here = os.path.dirname(os.path.abspath(__file__)) # search dir
top  = os.path.dirname(os.path.dirname(here)) # directory containing deephyper
sys.path.append(top)

MAX_QUEUED_TASKS = 128        # Limit of pending hyperparam evaluation jobs in DB
SERVICE_PERIOD = 2          # Delay (seconds) between main loop iterations
CHECKPOINT_INTERVAL = 10    # How many jobs to complete between optimizer checkpoints



def elapsed_timer(max_runtime_minutes=None):
    '''Iterator over elapsed seconds; ensure delay of SERVICE_PERIOD
    Raises StopIteration when time is up'''
    if max_runtime_minutes is None:
        max_runtime_minutes = float('inf')
        
    max_runtime = max_runtime_minutes * 60.0

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
    '''Command line parser'''
    parser = argparse.ArgumentParser()

    parser.add_argument("--benchmark", default='b1.addition_rnn',
                        help="name of benchmark module (e.g. b1.addition_rnn)"
                       )
    parser.add_argument("--backend", default='tensorflow',
                        help="Keras backend module name"
                       )
    parser.add_argument('--max-evals', type=int, default='10',
                        help='maximum number of evaluations'
                       )
    parser.add_argument('--from-checkpoint', default=None,
                        help='working directory of previous search, containing pickled optimizer'
                       )
    parser.add_argument('--repeat-evals', action='store_true',
                        help='Re-evaluate points visited by hyperparameter optimizer'
                       )
    return parser



def create_job(x, eval_counter, cfg):
    '''Add a new benchmark evaluation job to the Balsam DB'''

    jobname = f"task{eval_counter}"
    cmd = f"{sys.executable} {cfg.benchmark_filename}"
    args = ' '.join(f"--{p}={v}"
                    for p,v in zip(cfg.params, x) 
                    if 'hidden' not in p
                   )
    envs = f"KERAS_BACKEND={cfg.backend}"

    child = dag.spawn_child(
                name = jobname,
                direct_command = cmd,
                application_args = args,
                environ_vars = envs,
                wall_time_minutes = 2,
                num_nodes = 1, ranks_per_node = 1,
                threads_per_rank=64,
                wait_for_parents = False
               )
    print(f"Added task {eval_counter} to job DB")
    print(cmd, args)
    return child.job_id


def save_checkpoint(resultsList, opt_config, my_jobs, finished_jobs):
    '''Dump the current experiment state to disk'''
    print("checkpointing optimization")

    with open('optimizer.pkl', 'wb') as fp:
        pickle.dump(opt_config, fp)

    with open('jobs.json', 'w') as fp:
        jobsDict = dict(my_jobs=my_jobs, finished_jobs=finished_jobs)
        json.dump(jobsDict, fp, cls=Encoder)

    with open('results.json', 'w') as fp:
        json.dump(resultsList, fp, indent=4, sort_keys=True, cls=Encoder)

    keys = resultsList[0].keys() if resultsList else []
    with open('results.csv', 'w') as fp:
        dict_writer = csv.DictWriter(fp, keys)
        dict_writer.writeheader()
        dict_writer.writerows(resultsList)


def load_checkpoint(checkpoint_directory):
    '''Load the state of a previous run to resume experiment'''
    optpath = os.path.join(checkpoint_directory, 'optimizer.pkl')
    with open(optpath, 'rb') as fp:
        opt_config = pickle.load(fp)

    jobspath = os.path.join(checkpoint_directory, 'jobs.json')
    with open(jobspath, 'r') as fp:
        jobsDict = json.load(fp)
    my_jobs = jobsDict['my_jobs']
    finished_jobs = jobsDict['finished_jobs']
    
    resultpath = os.path.join(checkpoint_directory, 'results.json')
    with open(resultpath, 'r') as fp: 
        resultsList = json.load(fp)
    return opt_config, my_jobs, finished_jobs, resultsList
        

def next_points(cfg, eval_counter, my_jobs):
    '''Query optimizer for the next set of points to evaluate'''
    if cfg.starting_point is not None:
        XX = [cfg.starting_point]
        cfg.starting_point = None
    elif eval_counter < cfg.max_evals:
        already_active = BalsamJob.objects.filter(job_id__in=my_jobs.keys())
        already_active = already_active.exclude(state__in=END_STATES).count()
        print("Tracking", already_active, "pending jobs")
        num_tocreate = max(MAX_QUEUED_TASKS - already_active, 0)
        num_tocreate = min(num_tocreate, cfg.max_evals - eval_counter)
        XX = cfg.optimizer.ask(n_points=num_tocreate) if num_tocreate else []
    else:
        print("Reached max_evals; no longer starting new runs")
        XX = []

    if not cfg.repeat_evals:
        XX = [x for x in XX if json.dumps(x, cls=Encoder) not in my_jobs.values()]
    return XX


def read_cost(fname):
    '''Parse OUTPUT line from keras model fit'''
    cost = sys.float_info.max
    with open(fname, 'rt') as fp:
        for linenum, line in enumerate(fp):
            if "OUTPUT:" in line.upper():
                str1 = line.rstrip('\n')
                res = findall('OUTPUT:(.*)', str1)
                rv = float(res[0])
                if isnan(rv):
                    rv = sys.float_info.max
                cost = rv
                break
    return cost


def read_result(job, my_jobs):
    '''Return dict of hyperparams, cost, runtime'''
    outfile = os.path.join(job.working_directory, f"{job.name}.out")
    cost = read_cost(outfile)
    x = json.loads(my_jobs[job.job_id.hex])
    runtime = job.runtime_seconds
    result = dict(run_time=runtime, x=x, cost=cost)
    return result


def main():
    '''Service loop: add jobs; read results; drive optimizer'''

    # Initialize optimizer
    parser = create_parser()
    args = parser.parse_args()
    cfg = configureOptimizer(args)

    if dag.current_job is None:
        this = dag.add_job(name='search', workflow=args.benchmark,
                           wall_time_minutes=60
                          )
        this.create_working_path()
        this.update_state('JOB_FINISHED')
        dag.current_job = this
        dag.JOB_ID = this.job_id
        os.chdir(this.working_directory)
        print(f"Running in Balsam job directory: {this.working_directory}")

    walltime = dag.current_job.wall_time_minutes
    timer = elapsed_timer(max_runtime_minutes=walltime)
    eval_counter = 0
    chkpoint_counter = 0

    resultsList = []
    my_jobs = {}
    finished_jobs = []

    if args.from_checkpoint:
        chk_dir = args.from_checkpoint
    else:
        chk_dir = dag.current_job.working_directory

    if os.path.exists(os.path.join(chk_dir, 'optimizer.pkl')):
        cfg, my_jobs, finished_jobs, resultsList = load_checkpoint(chk_dir)
        eval_counter = len(my_jobs)
        print(f"Resume at eval # {eval_counter} from {chk_dir}")


    # Gracefully handle shutdown
    def handler(signum, stack):
        print('Received SIGINT/SIGTERM')
        save_checkpoint(resultsList, cfg, my_jobs, finished_jobs)
        sys.exit(0)

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)

    # MAIN LOOP
    print("Hyperopt driver starting")
    for elapsed_seconds in timer:
        print("\nElapsed time:", pretty_time(elapsed_seconds))
        if len(finished_jobs) == cfg.max_evals: break

        # Read in new results
        new_jobs = BalsamJob.objects.filter(job_id__in=my_jobs.keys())
        new_jobs = new_jobs.filter(state="JOB_FINISHED")
        new_jobs = new_jobs.exclude(job_id__in=finished_jobs)
        for job in new_jobs:
            try:
                result = read_result(job, my_jobs)
            except FileNotFoundError:
                print(f"ERROR: could not read output from {job.cute_id}")
            else:
                resultsList.append(result)
                print(f"Got data from {job.cute_id}")
                pprint(result)
                x, y = result['x'], result['cost']
                cfg.optimizer.tell(x, y)
                chkpoint_counter += 1
                if y == sys.float_info.max:
                    print(f"WARNING: {job.cute_id} cost was not found or NaN")
            finally:
                finished_jobs.append(job.job_id)
        
        # Which points are next?
        XX = next_points(cfg, eval_counter, my_jobs)
                
        # Create a BalsamJob for each point
        for x in XX:
            jobid = create_job(x, eval_counter, cfg)
            my_jobs[jobid.hex] = json.dumps(x, cls=Encoder)
            eval_counter += 1

        if chkpoint_counter >= CHECKPOINT_INTERVAL:
            save_checkpoint(resultsList, cfg, my_jobs, finished_jobs)
            chkpoint_counter = 0
        sys.stdout.flush()
    
    # EXIT
    print('Hyperopt driver finishing')
    save_checkpoint(resultsList, cfg, my_jobs, finished_jobs)

if __name__ == "__main__":
    main()

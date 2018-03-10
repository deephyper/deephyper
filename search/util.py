import argparse
import csv
import time
import logging

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
        print("next timer")
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
                        help='working directory of previous search, containing pickled optimizer'
                       )
    parser.add_argument('--evaluator', default='balsam')
    parser.add_argument('--repeat-evals', action='store_true',
                        help='Re-evaluate points visited by hyperparameter optimizer'
                       )
    return parser

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

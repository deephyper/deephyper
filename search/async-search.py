import json
import os
from pprint import pprint
import pickle
from re import findall
import signal
import sys

from deephyper.search import evaluate
from deephyper.search import util

HERE = os.path.dirname(os.path.abspath(__file__)) # search dir
top  = os.path.dirname(os.path.dirname(HERE)) # directory containing deephyper
sys.path.append(top)


SERVICE_PERIOD = 2          # Delay (seconds) between main loop iterations
CHECKPOINT_INTERVAL = 30    # How many jobs to complete between optimizer checkpoints


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
    timer = elapsed_timer(max_runtime_minutes=walltime,
                          service_period=SERVICE_PERIOD)
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
        print('top of service loop')
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
            except:
                raise
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
            print('exited create_job')
            my_jobs[jobid.hex] = json.dumps(x, cls=Encoder)
            print('added key to my_jobs')
            eval_counter += 1
        print('done with for x in XX')

        if chkpoint_counter >= CHECKPOINT_INTERVAL:
            print('trying to checkpoint')
            save_checkpoint(resultsList, cfg, my_jobs, finished_jobs)
            chkpoint_counter = 0
        print('skipping stdout flush')
        sys.stdout.flush()
    
    # EXIT
    print('Hyperopt driver finishing')
    save_checkpoint(resultsList, cfg, my_jobs, finished_jobs)

if __name__ == "__main__":
    main()

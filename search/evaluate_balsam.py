from deephyper.search import evaluate

import balsam.launcher.dag as dag
from balsam.service.models import BalsamJob, END_STATES

class BalsamEvaluator(evaluate.Evaluator):

    def __init__(self):
        super(self).__init__()
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

    def _submit_eval(self, x, cfg):
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

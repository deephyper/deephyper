from itertools import chain
from math import isnan
import time
import os
import sys

from importlib.util import find_spec
import logging

import balsam.launcher.dag as dag
from balsam.service.models import BalsamJob, END_STATES
from deephyper.evaluators import evaluate
logger = logging.getLogger(__name__)

from django.conf import settings


class BalsamEvaluator(evaluate.Evaluator):

    def __init__(self, params_list, bench_module_name, num_workers=1,
                 backend='tensorflow', model_path='', data_source='', 
                 stage_in_destination=''):
        super().__init__()

        self.id_key_map = {}
        self.num_workers = num_workers
        self.params_list = params_list
        self.bench_module_name = bench_module_name
        self.bench_file = os.path.abspath(find_spec(bench_module_name).origin)
        self.backend = backend
        self.model_path = model_path
        self.data_source = data_source
        self.stage_in_destination = stage_in_destination
        
        if dag.current_job is None:
            self._init_current_balsamjob()

        logger.debug("Balsam Evaluator instantiated")
        logger.debug(f"Backend: {self.backend}")
        logger.info(f"Benchmark: {self.bench_file}")

        if 'sqlite' in settings.DATABASES['default']['ENGINE']:
            logger.info('Detected sqlite backend; not using transacations')
        else:
            from django.db import transaction
            self.transaction_context = transaction.atomic
            logger.info('Detected NON-sqlite backend; using Transactions')

    def stop(self):
        pass

    def _init_current_balsamjob(self):
        this = dag.add_job(name='search', workflow=self.bench_module_name,
                           wall_time_minutes=60, state='JOB_FINISHED'
                          )
        workdir = this.working_directory
        if not os.path.exists(workdir): os.makedirs(workdir)
        dag.current_job = this
        dag.JOB_ID = this.job_id
        os.chdir(workdir)
        logger.debug(f"Running in Balsam job directory: {workdir}")

    def _eval_exec(self, x):
        jobname = f"task{self.counter}"
        cmd = f"{sys.executable} {self.bench_file}"
        
        param_dict = {p:v for p,v in zip(self.params_list, x)} # if 'hidden' not in p
        param_dict['model_path'] = self.model_path
        param_dict['data_source'] = self.data_source
        param_dict['stage_in_destination'] = self.stage_in_destination

        args = ' '.join(f"--{p}={v}" for p,v in param_dict.items())
        envs = f"KERAS_BACKEND={self.backend}"

        child = dag.add_job(
                    name = jobname,
                    workflow = dag.current_job.workflow,
                    direct_command = cmd,
                    application_args = args,
                    environ_vars = envs,
                    wall_time_minutes = 2,
                    num_nodes = 1, ranks_per_node = 1,
                    threads_per_rank=64,
                    serial_node_packing_count=2,
                    wait_for_parents = False,
                    state='PREPROCESSED'
                   )
        logger.debug(f"Created job {jobname}")
        logger.debug(f"Command: {cmd}")
        logger.debug(f"Args: {args}")

        self.id_key_map[child.job_id.hex] = self._encode(x)
        return child.job_id.hex

    def _read_eval_output(self, job):
        output = job.read_file_in_workdir(f'{job.name}.out')
        y = sys.float_info.max
        for line in output.split('\n'):
            if "OUTPUT:" in line.upper():
                y = float(line.split()[-1])
                break
        if isnan(y): y = sys.float_info.max
        return y

    def await_evals(self, to_read, timeout_sec=None, delay_sec=5):
        keys = [self._encode(x) for x in to_read]
        job_ids = [self.pending_evals[k] for k in keys
                   if k in self.pending_evals]
        job_ids = list(set(job_ids))
        assert all(k in self.evals for k in keys if k not in self.pending_evals)

        jobs = BalsamJob.objects.filter(job_id__in=job_ids)
        num_jobs = jobs.count()

        if timeout_sec and timeout_sec > delay_sec:
            num_checks = round(timeout_sec / delay_sec)
        else:
            num_checks = 1000000
        logger.info(f"Waiting on {num_jobs} Balsam evals to finish"
                    f" ({num_checks} checks with {delay_sec}s delay_sec)")
        checked_ids = []

        for i in range(num_checks):
            finished_jobs = jobs.filter(state='RUN_DONE')
            failed_jobs = jobs.filter(state__in=['RUN_ERROR', 'FAILED'])
            num_finished = finished_jobs.count()
            num_failed = failed_jobs.count()

            logger.debug(f"{num_finished+num_failed} out of {num_jobs} finished ({num_failed} failed)")
            isDone = (num_finished+num_failed) == num_jobs

            for job in finished_jobs.exclude(job_id__in=checked_ids):
                checked_ids.append(job.job_id.hex)
                key = self.id_key_map[job.job_id.hex]
                y = self._read_eval_output(job)
                self.evals[key] = y
                if key in self.pending_evals: del self.pending_evals[key]
                if key not in self.elapsed_times:
                    self.elapsed_times[key] = time.time() - self.start_seconds

            failed_ids = failed_jobs.exclude(job_id__in=checked_ids).values_list('job_id', flat=True)
            for job_id in failed_ids:
                checked_ids.append(job_id.hex)
                logger.warning(f"{job_id.hex} failed; marking objective as Inf")
                key = self.id_key_map[job_id.hex]
                y = sys.float_info.max
                self.evals[key] = y
                if key in self.pending_evals: del self.pending_evals[key]
                if key not in self.elapsed_times:
                    self.elapsed_times[key] = time.time() - self.start_seconds

            if isDone: break
            elif i < num_checks-1: time.sleep(delay_sec)

        for x, key in zip(to_read, keys):
            if key not in self.evals:
                logger.warning(f"Eval {key} never finished; marking infinity")
                if key in self.pending_evals:
                    jid = self.pending_evals[key]
                    job = BalsamJob.objects.get(job_id=jid)
                    dag.kill(job)
                    logger.info(f"Killed job {job.cute_id}")
                    del self.pending_evals[key]
                self.evals[key] = sys.float_info.max
                self.elapsed_times[key] = time.time() - self.start_seconds

            y = self.evals[key]
            logger.info(f"x: {x} y: {y}")
            yield (x,y)
    
    def get_finished_evals(self):
        '''iter over any immediately available results'''
        logger.info("Checking if any pending Balsam jobs are done")
        
        pending_ids = list(self.pending_evals.values())
        num_pending = len(pending_ids)
        num_blocks = 1 + (num_pending // 900)
        for i in range(num_blocks):
            query_block = pending_ids[i*900 : (i+1)*900]
            done_jobs = BalsamJob.objects.filter(job_id__in=query_block)
            done_jobs = done_jobs.filter(state='RUN_DONE')

            for job in done_jobs:
                logger.info(f"Got data from {job.cute_id}")
                key = self.id_key_map[job.job_id.hex]
                x = self._decode(key)
                y = self._read_eval_output(job)
                self.evals[key] = y
                if key not in self.elapsed_times:
                    self.elapsed_times[key] = time.time() - self.start_seconds
                del self.pending_evals[key]
                logger.info(f"x: {x} y: {y}")
                yield (x, y)
            
            error_jobs = BalsamJob.objects.filter(job_id__in=query_block)
            error_jobs = error_jobs.filter(state__in=['RUN_ERROR', 'FAILED', 'USER_KILLED'])

            for job in error_jobs:
                logger.info(f"Failed job: {job.cute_id}")
                key = self.id_key_map[job.job_id.hex]
                x = self._decode(key)
                y = sys.float_info.max
                self.evals[key] = y
                if key not in self.elapsed_times:
                    self.elapsed_times[key] = time.time() - self.start_seconds
                del self.pending_evals[key]
                logger.info(f"x: {x} y: {y}")
                yield (x, y)

        while self.repeated_evals:
            key = self.repeated_evals.pop()
            x = self._decode(key)
            if key in self.evals:
                y = self.evals[key]
                logger.info(f"giving repeated_eval x: {x} y: {y}")
                yield (x,y)
    
    def __getstate__(self):
        d = {}
        d['evals'] = self.evals
        d['pending_evals'] = self.pending_evals
        d['repeated_evals'] = self.repeated_evals
        d['id_key_map'] = self.id_key_map
        d['backend'] = self.backend
        d['num_workers'] = self.num_workers
        d['params_list'] = self.params_list
        d['bench_module_name'] = self.bench_module_name
        d['bench_file'] = self.bench_file
        return d

    def __setstate__(self, d):
        self.evals = d['evals']
        self.pending_evals = d['pending_evals']
        self.repeated_evals = d['repeated_evals']
        self.id_key_map = d['id_key_map']
        self.backend = d['backend']
        self.num_workers = d['num_workers']
        self.params_list = d['params_list']
        self.bench_module_name = d['bench_module_name']
        self.bench_file = d['bench_file']

        if dag.current_job is None:
            self._init_current_balsamjob()


        pending_ids = self.pending_evals.values()
        num_pending = BalsamJob.objects.filter(job_id__in=pending_ids).count()

        if num_pending != len(pending_ids):
            logger.error("Pickled Balsam evaluator had {len(pending_ids)} pending jobs")
            logger.error("But only {num_pending} found in Balsam DB")
            raise RuntimeError("Pending evals are missing in Balsam DB, did you delete them?")

        logger.debug("Balsam Evaluator loaded from pickle")
        logger.debug(f"Backend: {self.backend}")
        logger.info(f"Benchmark: {self.bench_file}")

        logger.info(f"Restored {len(self.evals)} finished evals")
        logger.info(f"Resuming {len(self.pending_evals)} evals")

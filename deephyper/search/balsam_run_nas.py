import balsam.launcher.dag as dag
import sys
import logging

logger = logging.getLogger(__name__)

jobname = "run_mnistNas"
path_run_nas = "/Users/dipendra/Projects/deephyper/search/run_nas.py"
cmd = f"{sys.executable} {path_run_nas}"
args = "--benchmark=cifar10Nas --run_module_name=model.nas --num-workers=1"

child = dag.add_job(
        name = jobname,
        workflow = jobname,
        direct_command = cmd,
        application_args = args,
        wall_time_minutes = 2,
        num_nodes = 1,
        ranks_per_node = 1,
        threads_per_rank=1,
        )

logger.debug(f"Created job {jobname}")
logger.debug(f"Command: {cmd}")
logger.debug(f"Args: {args}")

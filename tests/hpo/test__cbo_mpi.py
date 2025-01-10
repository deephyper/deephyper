import os
import sys
import time

import pytest
import deephyper.tests as dht

from deephyper.hpo import HpProblem, CBO
from deephyper.evaluator import Evaluator, JobStatus, profile

PYTHON = sys.executable
SCRIPT = os.path.abspath(__file__)


@profile
def run_sync(job):
    while True:
        time.sleep(0.1)
        if job.status is JobStatus.CANCELLING:
            break
    return job.parameters["x"]


def _test_mpi_timeout(tmp_path):
    """Test if the timeout condition is working properly when the run-function runs indefinitely."""
    problem = HpProblem()
    problem.add_hyperparameter((0.0, 10.0), "x")

    with Evaluator.create(run_sync, method="mpicomm") as evaluator:
        if evaluator.is_master:
            search = CBO(
                problem, evaluator, random_state=42, surrogate_model="DUMMY", log_dir=tmp_path
            )
            t1 = time.time()
            results = search.search(timeout=1)

            assert len(results) == evaluator.num_workers
            assert (results["job_status"] == "CANCELLED").all()

    # The search should have been interrupted after 1 second.
    # The following must be placed after exiting the context manager.
    if evaluator.is_master:
        duration = time.time() - t1
        print(f"DEEPHYPER-OUTPUT: {duration}")


@pytest.mark.mpi
def test_mpi_timeout(tmp_path):
    command = f"mpirun -np 4 {PYTHON} {SCRIPT} _test_mpi_timeout {tmp_path}"
    result = dht.run(command, live_output=False, timeout=5)
    duration = dht.parse_result(result.stdout)
    assert result.returncode == 0
    assert duration < 2


if __name__ == "__main__":
    func = sys.argv[-2]
    func = globals()[func]
    func(sys.argv[-1])

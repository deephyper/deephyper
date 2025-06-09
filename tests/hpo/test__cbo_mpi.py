import os
import sys
import time

import numpy as np
import pytest

import deephyper.tests as dht
from deephyper.evaluator import Evaluator, JobStatus, profile
from deephyper.evaluator.callback import TqdmCallback
from deephyper.hpo import CBO, HpProblem

PYTHON = sys.executable
SCRIPT = os.path.abspath(__file__)


@profile
def run_sync_sleep_forever(job):
    while True:
        time.sleep(0.1)
        if job.status is JobStatus.CANCELLING:
            break
    return job.parameters["x"]


def _test_mpi_timeout(tmp_path):
    """Test if the timeout condition is working properly when the run-function runs indefinitely."""
    problem = HpProblem()
    problem.add_hyperparameter((0.0, 10.0), "x")

    with Evaluator.create(run_sync_sleep_forever, method="mpicomm") as evaluator:
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
    evaluator.close()


@pytest.mark.mpi
def test_mpi_timeout(tmp_path):
    command = f"mpirun -np 4 {PYTHON} {SCRIPT} _test_mpi_timeout {tmp_path}"
    result = dht.run(command, live_output=False, timeout=5)
    duration = dht.parse_result(result.stdout)
    assert result.returncode == 0
    assert duration < 2


@profile
def run(job):
    from deephyper.evaluator.mpi import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    return {"objective": job.parameters["x"], "metadata": {"rank": rank}}


def _test_mpi_many_initial_points(tmp_path):
    """Test submitting many initial points."""
    problem = HpProblem()
    problem.add_hyperparameter((0.0, 10.0), "x")

    max_evals = 100

    with Evaluator.create(
        run, method="mpicomm", method_kwargs={"callbacks": [TqdmCallback()]}
    ) as evaluator:
        if evaluator.is_master:
            print(f"{evaluator.num_workers=}")
            search = CBO(
                problem,
                evaluator,
                random_state=42,
                surrogate_model="DUMMY",
                log_dir=tmp_path,
                initial_points=[{"x": v} for v in np.linspace(0.0, 10.0, max_evals)],
            )
            results = search.search(max_evals, max_evals_strict=True)

            assert len(results) == max_evals
    if evaluator.is_master:
        print("DEEPHYPER-OUTPUT: 0")
    evaluator.close()


@pytest.mark.mpi
def test_mpi_many_initial_points(tmp_path):
    command = f"mpirun -np 4 {PYTHON} {SCRIPT} _test_mpi_many_initial_points {tmp_path}"
    result = dht.run(command, live_output=False, timeout=20)
    status = dht.parse_result(result.stdout)
    assert result.returncode == 0
    assert status == 0


if __name__ == "__main__":
    func = sys.argv[-2]
    func = globals()[func]
    func(sys.argv[-1])

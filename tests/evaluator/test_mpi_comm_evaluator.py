import os
import sys
import time

import pytest

PYTHON = sys.executable
SCRIPT = os.path.abspath(__file__)

import deephyper.test  # noqa: E402
from deephyper.evaluator import Evaluator  # noqa: E402


def run(job):
    job_id = int(job.id.split(".")[1])

    print(f"job {job_id}...")

    if job_id > 3:
        frac = 0.1
        cum = 0
        while cum < 4:
            print(job)
            time.sleep(frac)
            cum += frac

    print(f"job {job_id} done!", flush=True)

    return job.parameters["x"]


def _test_mpicomm_evaluator():
    """Test the MPICommEvaluator"""

    configs = [{"x": i} for i in range(8)]

    t1 = time.time()
    with Evaluator.create(
        run, method="mpicomm", method_kwargs={"abort_on_exit": False}
    ) as evaluator:
        if evaluator is not None:
            print(configs)
            evaluator.submit(configs)

            results = evaluator.gather(type="BATCH", size=4)
            print("gather", flush=True)
            objectives = sorted([job.output for job in results])
            assert objectives == list(range(4))
    duration = time.time() - t1

    from mpi4py import MPI  # noqa: E402

    rank = MPI.COMM_WORLD.Get_rank()
    print(f"{rank=}, {duration=}")


@pytest.mark.mpi
def test_mpicomm_evaluator():
    command = f"mpirun -np 4 {PYTHON} {SCRIPT} _test_mpicomm_evaluator"
    result = deephyper.test.run(command, live_output=False)
    assert result.returncode == 0


if __name__ == "__main__":
    func = sys.argv[-1]
    func = globals()[func]
    func()

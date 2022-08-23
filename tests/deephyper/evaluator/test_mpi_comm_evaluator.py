import os
from re import A
import sys
import time

import pytest

PYTHON = sys.executable
SCRIPT = os.path.abspath(__file__)

import deephyper.test
from deephyper.evaluator import Evaluator


def run(config):
    job_id = config["job_id"]
    print(f"job {job_id}...")
    if job_id > 3:
        time.sleep(10)
    print(f"job {job_id} done!", flush=True)
    return config["x"]


def _test_mpicomm_evaluator():
    """Test the MPICommEvaluator"""

    configs = [{"x": i} for i in range(8)]

    with Evaluator.create(
        run, method="mpicomm", method_kwargs={"abort_on_exit": True}
    ) as evaluator:
        if evaluator is not None:
            print(configs)
            evaluator.submit(configs)

            results = evaluator.gather(type="BATCH", size=4)
            print("gather", flush=True)
            objectives = sorted([job.result for job in results])
            assert objectives == list(range(4))


@pytest.mark.fast
@pytest.mark.hps
@pytest.mark.mpi
def test_mpicomm_evaluator():
    command = f"mpirun -np 4 {PYTHON} {SCRIPT} _test_mpicomm_evaluator"
    result = deephyper.test.run(command, live_output=False)


if __name__ == "__main__":
    func = sys.argv[-1]
    func = globals()[func]
    func()

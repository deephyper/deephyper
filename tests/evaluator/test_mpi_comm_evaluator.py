import os
import sys
import time

import pytest

PYTHON = sys.executable
SCRIPT = os.path.abspath(__file__)

import deephyper.tests as dht  # noqa: E402
from deephyper.evaluator import Evaluator, profile  # noqa: E402


@profile
def run_sync(job):
    job_id = int(job.id.split(".")[1])

    if job_id > 3:
        frac = 0.1
        for i in range(5):
            dht.log(f"{job.id=} doing stuff...")
            time.sleep(frac)

    return job.parameters["x"]


def _test_mpicomm_evaluator():
    """Test the MPICommEvaluator"""
    from mpi4py import MPI  # noqa: E402

    if not MPI.Is_initialized():
        MPI.Init_thread()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    configs = [{"x": i} for i in range(8)]

    t1 = time.time()

    with Evaluator.create(
        run_sync,
        method="mpicomm",
    ) as evaluator:
        dht.log(f"{rank=} {evaluator=}")
        if evaluator.is_master:
            assert evaluator.num_workers == size - 1

            evaluator.submit(configs)

            results = evaluator.gather(type="BATCH", size=4)
            results += evaluator.gather(type="BATCH", size=4)

            assert len(results) == 8
            assert all("timestamp_submit" in j.metadata for j in results)
            assert all("timestamp_gather" in j.metadata for j in results)
            assert all("timestamp_start" in j.metadata for j in results)
            assert all("timestamp_end" in j.metadata for j in results)

            objectives = sorted([job.output for job in results])
            assert objectives == list(range(8))

            duration = time.time() - t1
            assert duration < 2

            durations = [
                j.metadata["timestamp_end"] - j.metadata["timestamp_start"]
                for j in results
                if int(j.id.split(".")[1]) <= 3
            ]
            dht.log(f"{rank=} {durations=}")
            assert len(durations) == 4 and all(d < 0.1 for d in durations)
            durations = [
                (j.metadata["timestamp_end"] - j.metadata["timestamp_start"])
                for j in results
                if int(j.id.split(".")[1]) > 3
            ]
            assert len(durations) == 4 and all(d > 0.5 for d in durations)

    # Test that all process terminate properly
    comm.Barrier()


@pytest.mark.mpi
def test_mpicomm_evaluator():
    command = f"mpirun -np 4 {PYTHON} {SCRIPT} _test_mpicomm_evaluator"
    result = dht.run(command, live_output=False, timeout=5)
    assert result.returncode == 0


if __name__ == "__main__":
    func = sys.argv[-1]
    func = globals()[func]
    func()

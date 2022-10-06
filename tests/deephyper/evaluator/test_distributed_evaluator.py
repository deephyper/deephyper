"""
mpirun -np 2 python test_distributed_evaluator.py
"""
import os
import sys
import time

PYTHON = sys.executable
SCRIPT = os.path.abspath(__file__)

import pytest

import deephyper.test
from deephyper.evaluator._serial import SerialEvaluator
from deephyper.evaluator._distributed import distributed


def run(config):
    r = config["r"]
    if r == 1:
        time.sleep(2)
    print(f"r={r}")
    return config["r"]


def _test_mpi_distributed_evaluator():
    evaluator = distributed(backend="mpi")(SerialEvaluator)(run)

    configs = [{"i": i, "r": evaluator.rank} for i in range(1)]

    # test synchronous share
    evaluator.submit(configs)
    results = evaluator.gather("ALL", sync_communication=True)

    if evaluator.rank == 0:
        print(f"{results=}", flush=True)
    assert len(results) == evaluator.size * len(configs)

    # test asynchronous share
    evaluator.submit(configs)
    results = evaluator.gather("ALL")

    print(f"r={evaluator.rank} -> {len(results)}")
    if evaluator.rank != 1:
        assert len(results) <= (evaluator.size - 1) * len(configs)
    else:
        assert len(results) == (evaluator.size) * len(configs)


@pytest.mark.fast
@pytest.mark.hps
@pytest.mark.mpi
def test_mpi_distributed_evaluator():
    command = f"mpirun -np 4 {PYTHON} {SCRIPT} _test_mpi_distributed_evaluator"
    result = deephyper.test.run(command, live_output=False)


if __name__ == "__main__":
    func = sys.argv[-1]
    func = globals()[func]
    func()

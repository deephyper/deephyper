import os
import sys
import time

import pytest

PYTHON = sys.executable
SCRIPT = os.path.abspath(__file__)

import deephyper.test


def _test_random_search_mpicomm():
    """Example to execute:

    mpirun -np 4 python test_random_mpicomm.py
    """

    from deephyper.evaluator import Evaluator
    from deephyper.nas.run import run_debug_slow
    from deephyper.search.nas import Random
    from deephyper.test.nas import linearReg

    with Evaluator.create(run_debug_slow, method="mpicomm") as evaluator:
        if evaluator:

            search = Random(
                linearReg.Problem,
                evaluator,
                log_dir="log-random-mpicomm",
                random_state=42,
            )
            t1 = time.time()
            res = search.search(timeout=2)
            duration = time.time() - t1

            assert len(res) >= 1
            assert duration < 3


@pytest.mark.slow
@pytest.mark.nas
@pytest.mark.mpi
def test_mpicomm_evaluator():
    command = f"mpirun -np 4 {PYTHON} {SCRIPT} _test_random_search_mpicomm"
    result = deephyper.test.run(command, live_output=True)


if __name__ == "__main__":
    func = sys.argv[-1]
    func = globals()[func]
    func()

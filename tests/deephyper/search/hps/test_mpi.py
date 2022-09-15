import os
import sys
import pytest

PYTHON = sys.executable
SCRIPT = os.path.abspath(__file__)

import deephyper.test


def _test_mpi_timeout():
    import time
    from deephyper.problem import HpProblem
    from deephyper.search.hps import CBO
    from deephyper.evaluator import Evaluator

    problem = HpProblem()
    problem.add_hyperparameter((0.0, 10.0), "x")

    def run(config):
        while True:
            1 + 1
            time.sleep(0.1)
        return config["x"]

    with Evaluator.create(run, method="mpicomm") as evaluator:

        if evaluator:
            search = CBO(problem, run, random_state=42, surrogate_model="DUMMY")
            search.search(timeout=1)


@pytest.mark.fast
@pytest.mark.hps
@pytest.mark.mpi
def test_mpi_timeout():
    command = f"time mpirun -np 4 {PYTHON} {SCRIPT} _test_mpi_timeout"
    result = deephyper.test.run(command, live_output=False)
    result = result.stderr.replace("\n", "").split(" ")
    i = result.index("sys")
    t = float(result[i-1])
    assert t < 2


if __name__ == "__main__":
    func = sys.argv[-1]
    func = globals()[func]
    func()

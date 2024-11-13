import os
import sys
import pytest

PYTHON = sys.executable
SCRIPT = os.path.abspath(__file__)

import deephyper.test


def _test_mpi_timeout(tmp_path):
    """Test if the timeout condition is working properly when the run-function runs indefinitely."""
    import time
    from deephyper.hpo import HpProblem
    from deephyper.hpo import CBO
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
            search = CBO(
                problem, run, random_state=42, surrogate_model="DUMMY", log_dir=tmp_path
            )
            t1 = time.time()
            search.search(timeout=1)

    # The search should have been interrupted after 1 second.
    # The following must be placed after exiting the context manager.
    if evaluator:
        duration = time.time() - t1
        print(f"DEEPHYPER-OUTPUT: {duration}")


@pytest.mark.fast
@pytest.mark.mpi
def test_mpi_timeout(tmp_path):
    command = f"mpirun -np 4 {PYTHON} {SCRIPT} _test_mpi_timeout {tmp_path}"
    result = deephyper.test.run(command, live_output=False)
    duration = deephyper.test.parse_result(result.stdout)
    assert duration < 2


if __name__ == "__main__":
    func = sys.argv[-2]
    func = globals()[func]
    func(sys.argv[-1])

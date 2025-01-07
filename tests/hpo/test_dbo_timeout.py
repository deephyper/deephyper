import os
import shutil
import sys
import pytest

PYTHON = sys.executable
SCRIPT = os.path.abspath(__file__)

import deephyper.tests


def _test_dbo_timeout():
    import time
    import numpy as np

    from deephyper.hpo import HpProblem
    from deephyper.hpo import MPIDistributedBO

    d = 10
    domain = (-32.768, 32.768)
    hp_problem = HpProblem()
    for i in range(d):
        hp_problem.add_hyperparameter(domain, f"x{i}")

    def ackley(x, a=20, b=0.2, c=2 * np.pi):
        d = len(x)
        s1 = np.sum(x**2)
        s2 = np.sum(np.cos(c * x))
        term1 = -a * np.exp(-b * np.sqrt(s1 / d))
        term2 = -np.exp(s2 / d)
        y = term1 + term2 + a + np.exp(1)
        return y

    def run(job):
        config = job.parameters
        x = np.array([config[f"x{i}"] for i in range(d)])
        x = np.asarray_chkfinite(x)  # ValueError if any NaN or Inf
        return -ackley(x)

    log_dir = "log-dbo"
    search = MPIDistributedBO(
        hp_problem,
        run,
        log_dir=log_dir,
    )

    timeout = 2
    if search.rank == 0:
        t1 = time.time()
        results = search.search(timeout=timeout)
        duration = time.time() - t1
        assert duration < timeout + 1
    else:
        search.search(timeout=timeout)
    search.comm.Barrier()
    if search.rank == 0:
        print("\n", results)
        shutil.rmtree(log_dir)


@pytest.mark.mpi
@pytest.mark.redis
def test_dbo_timeout():
    command = f"time mpirun -np 4 {PYTHON} {SCRIPT} _test_dbo_timeout"
    result = deephyper.test.run(command, live_output=False)
    result = result.stderr.replace("\n", "").split(" ")
    i = result.index("sys")
    t = float(result[i - 1])
    assert t < 3


if __name__ == "__main__":
    func = sys.argv[-1]
    func = globals()[func]
    func()

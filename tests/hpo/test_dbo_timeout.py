import multiprocessing
import os
import sys
import pytest

import deephyper.tests as dht

PYTHON = sys.executable
SCRIPT = os.path.abspath(__file__)
CPUS = min(4, multiprocessing.cpu_count())


def _test_dbo_timeout(tmp_path):
    import time
    import numpy as np

    from deephyper.evaluator.mpi import MPI
    from deephyper.hpo import HpProblem, MPIDistributedBO

    if not MPI.Is_initialized():
        MPI.Init_thread()
    rank = int(MPI.COMM_WORLD.Get_rank())

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
        return {"objective": -ackley(x), "metadata": {"rank": rank}}

    search = MPIDistributedBO(
        hp_problem,
        run,
        surrogate_model="DUMMY",
        log_dir=tmp_path,
    )

    timeout = 2
    t1 = time.time()
    if search.rank == 0:
        results = search.search(timeout=timeout)
    else:
        search.search(timeout=timeout)

    duration = time.time() - t1
    assert duration < timeout + 1
    if search.rank == 0:
        assert len(results) > 1

    search.comm.Barrier()


@pytest.mark.mpi
@pytest.mark.redis
def test_dbo_timeout(tmp_path):
    command = f"mpirun -np {CPUS} {PYTHON} {SCRIPT} _test_dbo_timeout {tmp_path}"
    result = dht.run(command, live_output=False, timeout=10)
    assert result.returncode == 0


if __name__ == "__main__":
    func = sys.argv[-2]
    func = globals()[func]
    func(sys.argv[-1])

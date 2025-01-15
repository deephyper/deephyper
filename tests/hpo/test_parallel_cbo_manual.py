import multiprocessing
import os
import sys
import pytest

import deephyper.tests as dht

PYTHON = sys.executable
SCRIPT = os.path.abspath(__file__)
CPUS = min(4, multiprocessing.cpu_count())


def _test_parallel_cbo_manual(tmp_path):
    from deephyper.evaluator import ThreadPoolEvaluator
    from deephyper.evaluator.mpi import MPI
    from deephyper.evaluator.storage import RedisStorage
    from deephyper.hpo import HpProblem, CBO

    if not MPI.Is_initialized():
        MPI.Init_thread()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    def run(job):
        return -(job.parameters["x"] ** 2)

    problem = HpProblem()
    problem.add_hyperparameter((-10.0, 10.0), "x")

    storage = RedisStorage()
    storage.connect()
    storage._redis.flushdb()

    search_id = None
    if rank == 0:
        evaluator = ThreadPoolEvaluator(run, storage=storage)
        search = CBO(problem, evaluator, random_state=42, log_dir=tmp_path)
        search_id = search.search_id
        print(f"{search_id}")

    search_id = comm.bcast(search_id)
    print(f"rank={rank} - search_id={search_id}")

    if rank > 0:
        evaluator = ThreadPoolEvaluator(run, storage=storage, search_id=search_id)
        search = CBO(problem, evaluator, surrogate_model="DUMMY", random_state=42)
        search.is_master = False
    comm.Barrier()

    if rank == 0:
        results = search.search(max_evals=20)
    else:
        search.search(max_evals=20)
    comm.Barrier()

    if rank == 0:
        print(f"{len(results)} results")
        print(results.objective.tolist())
    comm.Barrier()


@pytest.mark.mpi
@pytest.mark.redis
def test_parallel_cbo_manual(tmp_path):
    command = f"mpirun -np {CPUS} {PYTHON} {SCRIPT} _test_parallel_cbo_manual {tmp_path}"
    result = dht.run(command, live_output=False, timeout=10)
    assert result.returncode == 0


if __name__ == "__main__":
    func = sys.argv[-2]
    func = globals()[func]
    func(sys.argv[-1])

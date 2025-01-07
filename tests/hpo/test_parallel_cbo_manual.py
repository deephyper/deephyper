import os
import shutil
import sys
import pytest

PYTHON = sys.executable
SCRIPT = os.path.abspath(__file__)

import deephyper.tests


def _test_parallel_cbo_manual():

    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    from deephyper.hpo import HpProblem

    def run(job):
        return -(job.parameters["x"] ** 2)

    from deephyper.hpo import CBO
    from deephyper.evaluator import SerialEvaluator
    from deephyper.evaluator.storage import RedisStorage

    problem = HpProblem()
    problem.add_hyperparameter((-10.0, 10.0), "x")

    storage = RedisStorage()
    storage.connect()
    storage._redis.flushdb()

    search_id = None
    if rank == 0:
        evaluator = SerialEvaluator(run, storage=storage)
        search = CBO(problem, evaluator, random_state=42)
        search_id = search.search_id
        print(f"{search_id}")

    search_id = comm.bcast(search_id)
    print(f"rank={rank} - search_id={search_id}")

    if rank > 0:
        evaluator = SerialEvaluator(run, storage=storage, search_id=search_id)

        def dumps_evals(*args, **kwargs):
            pass

        evaluator.dump_jobs_done_to_csv = dumps_evals

        search = CBO(problem, evaluator, random_state=42)
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
def test_dbo_timeout():
    command = f"time mpirun -np 4 {PYTHON} {SCRIPT} _test_parallel_cbo_manual"
    result = deephyper.test.run(command, live_output=False)
    result = result.stderr.replace("\n", "").split(" ")
    # i = result.index("sys")
    # t = float(result[i - 1])
    # assert t < 3


if __name__ == "__main__":
    func = sys.argv[-1]
    func = globals()[func]
    func()

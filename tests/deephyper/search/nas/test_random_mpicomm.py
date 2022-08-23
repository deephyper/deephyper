import pytest


@pytest.mark.fast
@pytest.mark.nas
@pytest.mark.mpi
def test_random_search_mpicomm():
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
            res = search.search(timeout=2)


if __name__ == "__main__":
    test_random_search_mpicomm()

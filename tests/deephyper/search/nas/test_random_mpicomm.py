import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


@pytest.mark.nas
def test_random_search_mpicomm():
    """Example to execute:

    mpirun -np 4 python test_random_mpicomm.py
    """

    from deephyper.evaluator import Evaluator
    from deephyper.nas.run import run_debug_slow
    from deephyper.search.nas import Random

    import problems.linearReg as linearReg

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

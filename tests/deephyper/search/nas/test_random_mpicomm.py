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
    from deephyper.nas.run import run_debug_arch, run_debug_slow
    from deephyper.search.nas import Random

    import linearReg

    with Evaluator.create(run_debug_slow, method="mpicomm") as evaluator:
        if evaluator:
            import logging

            logging.basicConfig(
                # filename=path_log_file, # optional if we want to store the logs to disk
                level=logging.INFO,
                format="%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s - %(message)s",
                force=True,
            )

            search = Random(
                linearReg.Problem,
                evaluator,
                log_dir="log-random-mpicomm",
                random_state=42,
            )
            res = search.search(timeout=2)
        #     print("master done")
        # print("worker done")


if __name__ == "__main__":
    test_random_search_mpicomm()

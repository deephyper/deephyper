import logging
import pathlib
from deephyper.evaluator import Evaluator


def run(config):
    return config["x"]


def _test_mpicomm_evaluator():
    """Test the MPICommEvaluator"""

    configs = [{"x": i} for i in range(4)]

    with Evaluator.create(
        run,
        method="mpicomm",
    ) as evaluator:
        if evaluator is not None:
            evaluator.submit(configs)

            results = evaluator.gather("ALL")
            objectives = sorted([job.result for job in results])
            assert objectives == list(range(4))


if __name__ == "__main__":
    _test_mpicomm_evaluator()

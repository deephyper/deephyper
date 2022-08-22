import time
from deephyper.evaluator import Evaluator


def run(config):
    if config["job_id"] > 4:
        time.sleep(3)
    print(f"call: {config}")
    return config["x"]


def _test_mpicomm_evaluator():
    """Test the MPICommEvaluator"""

    configs = [{"x": i} for i in range(8)]

    with Evaluator.create(
        run,
        method="mpicomm",
    ) as evaluator:
        if evaluator is not None:
            evaluator.submit(configs)

            results = evaluator.gather(type="BATCH", size=4)
            objectives = sorted([job.result for job in results])
            assert objectives == list(range(4))


if __name__ == "__main__":
    _test_mpicomm_evaluator()

import unittest
import pytest

from deephyper.evaluator import Evaluator, RunningJob
from deephyper.evaluator.storage._null_storage import NullStorage


def run(job: RunningJob) -> dict:
    return job.parameters["x"] + 1


@pytest.mark.fast
class TestNullStorage(unittest.TestCase):
    def test_basic(self):

        evaluator = Evaluator.create(
            run,
            method="serial",
            method_kwargs={
                "storage": NullStorage(),
            },
        )

        evaluator.submit([{"x": i} for i in range(10)])
        jobs_done = evaluator.gather("ALL")
        inputs = list(map(lambda job: job.args, jobs_done))
        results = list(map(lambda job: job.output, jobs_done))
        print(f"{inputs=}")
        print(f"{results=}")


if __name__ == "__main__":
    TestNullStorage().test_basic()

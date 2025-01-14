from deephyper.evaluator import Evaluator, RunningJob
from deephyper.evaluator.storage._null_storage import NullStorage


async def run(job: RunningJob) -> dict:
    return job.parameters["x"] + 1


def test_basic():
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
    evaluator.close()

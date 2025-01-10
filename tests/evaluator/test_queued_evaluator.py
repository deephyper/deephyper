import pytest


async def run_async(config, dequed=None):
    return config["x"] + dequed[0]


def run_sync(config, dequed=None):
    return config["x"] + dequed[0]


def test_queued_serial_evaluator():
    from deephyper.evaluator import SerialEvaluator, queued, HPOJob

    QueuedSerialEvaluator = queued(SerialEvaluator)  # returns class of type Queued{evaluator_class}

    evaluator = QueuedSerialEvaluator(
        run_async,
        num_workers=1,
        # queued arguments
        queue=[1, 2, 3, 4],
        queue_pop_per_task=1,
    )
    evaluator._job_class = HPOJob

    assert evaluator.num_workers == 1
    assert list(evaluator.queue) == [1, 2, 3, 4]
    assert evaluator.queue_pop_per_task == 1

    results = []
    for i in range(8):
        evaluator.submit([{"x": 0}])

        jobs = evaluator.gather("ALL")

        results.append(jobs[0].objective)

    assert results == [1, 2, 3, 4, 1, 2, 3, 4]

    evaluator.close()


@pytest.mark.ray
def test_queued_ray_evaluator():
    try:
        import os

        HERE = os.path.dirname(os.path.abspath(__file__))

        from deephyper.evaluator import RayEvaluator, queued, HPOJob

        QueuedRayEvaluator = queued(RayEvaluator)  # returns class of type Queued{evaluator_class}

        evaluator = QueuedRayEvaluator(
            run_sync,
            num_cpus=1,
            num_cpus_per_task=1,
            num_workers=1,
            ray_kwargs={"runtime_env": {"working_dir": HERE}},
            # queued arguments
            queue=[1, 2, 3, 4],
            queue_pop_per_task=1,
        )
        evaluator._job_class = HPOJob

        assert evaluator.num_workers == 1
        assert list(evaluator.queue) == [1, 2, 3, 4]
        assert evaluator.queue_pop_per_task == 1

        results = []
        for i in range(8):
            evaluator.submit([{"x": 0}])

            jobs = evaluator.gather("ALL")

            results.append(jobs[0].result)

        assert results == [1, 2, 3, 4, 1, 2, 3, 4]
    except ImportError as e:
        e_str = str(e)
        if "RayEvaluator" not in e_str:
            raise e

    evaluator.close()

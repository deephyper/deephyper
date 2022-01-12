import pytest

def run(config, dequed=None):
    return config["x"] + dequed[0]


class TestQueuedEvaluator:
    def test_queued_ray_evaluator(self):
        from deephyper.evaluator import RayEvaluator
        from deephyper.evaluator import queued

        QueuedRayEvaluator = queued(RayEvaluator) # returns class of type Queued{evaluator_class}

        evaluator = QueuedRayEvaluator(
            run,
            num_cpus=1, 
            num_cpus_per_task=1,
            num_workers=1,
            # queued arguments 
            queue=[1,2,3,4], 
            queue_pop_per_task=1)

        assert evaluator.num_workers == 1
        assert list(evaluator.queue) == [1,2,3,4]
        assert evaluator.queue_pop_per_task == 1

        results = []
        for i in range(8):
            evaluator.submit([{"x": 0}])

            jobs = evaluator.gather("ALL")

            results.append(jobs[0].result)

        assert results == [1,2,3,4,1,2,3,4]
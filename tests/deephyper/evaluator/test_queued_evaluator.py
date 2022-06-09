import pytest
import unittest


def run(config, dequed=None):
    return config["x"] + dequed[0]


class TestQueuedEvaluator(unittest.TestCase):
    @pytest.mark.hps_fast_test
    def test_queued_serial_evaluator(self):
        from deephyper.evaluator import SerialEvaluator, queued

        QueuedSerialEvaluator = queued(
            SerialEvaluator
        )  # returns class of type Queued{evaluator_class}

        evaluator = QueuedSerialEvaluator(
            run,
            num_workers=1,
            # queued arguments
            queue=[1, 2, 3, 4],
            queue_pop_per_task=1,
        )

        assert evaluator.num_workers == 1
        assert list(evaluator.queue) == [1, 2, 3, 4]
        assert evaluator.queue_pop_per_task == 1

        results = []
        for i in range(8):
            evaluator.submit([{"x": 0}])

            jobs = evaluator.gather("ALL")

            results.append(jobs[0].result)

        assert results == [1, 2, 3, 4, 1, 2, 3, 4]

    @pytest.mark.ray
    def test_queued_ray_evaluator(self):
        try:
            from deephyper.evaluator import RayEvaluator, queued

            QueuedRayEvaluator = queued(
                RayEvaluator
            )  # returns class of type Queued{evaluator_class}

            evaluator = QueuedRayEvaluator(
                run,
                num_cpus=1,
                num_cpus_per_task=1,
                num_workers=1,
                # queued arguments
                queue=[1, 2, 3, 4],
                queue_pop_per_task=1,
            )

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
            if not ("RayEvaluator" in e_str):
                raise e

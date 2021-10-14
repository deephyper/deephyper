import pytest
from deephyper.core.exceptions import DeephyperRuntimeError


def run(config):
    return config["x"]


@pytest.mark.incremental
class TestEvaluator:
    def test_import(self):
        from deephyper.evaluator import Evaluator

    def test_wrong_evaluator(self):
        from deephyper.evaluator import Evaluator

        with pytest.raises(DeephyperRuntimeError):
            evaluator = Evaluator.create(
                run,
                method="threadPool",
                method_kwargs={
                    "num_workers": 1,
                },
            )

    def test_thread_process_subprocess(self):
        from deephyper.evaluator import Evaluator

        for method in ["thread", "process", "subprocess"]:
            evaluator = Evaluator.create(
                run,
                method=method,
                method_kwargs={
                    "num_workers": 1,
                },
            )

            configs = [{"x": i} for i in range(10)]
            evaluator.submit(configs)
            jobs = evaluator.gather("ALL")
            jobs.sort(key=lambda j: j.config["x"])
            for config,job in zip(configs, jobs):
                assert config["x"] == job.config["x"]

            evaluator.submit(configs)
            jobs = evaluator.gather("BATCH", size=1)
            assert 1 <= len(jobs) and len(jobs) <= len(configs)

    def test_ray(self):
        from deephyper.evaluator import Evaluator

        def run(config):
            return config["x"]

        evaluator = Evaluator.create(
            run,
            method="ray",
            method_kwargs={
                "num_cpus": 1,
            },
        )

        configs = [{"x": i} for i in range(10)]
        evaluator.submit(configs)
        jobs = evaluator.gather("ALL")
        jobs.sort(key=lambda j: j.config["x"])
        for config,job in zip(configs, jobs):
            assert config["x"] == job.config["x"]

        evaluator.submit(configs)
        jobs = evaluator.gather("BATCH", size=1)
        assert 1 <= len(jobs) and len(jobs) <= len(configs)

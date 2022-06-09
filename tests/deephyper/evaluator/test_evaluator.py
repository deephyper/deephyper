import unittest
import pytest


def run(config, y=0):
    return config["x"] + y


def run_many_results(config, y=0):
    return {"x": config["x"], "y": y}


class TestEvaluator(unittest.TestCase):
    @pytest.mark.fast
    def test_import(self):
        from deephyper.evaluator import Evaluator

    @pytest.mark.fast
    def test_wrong_evaluator(self):
        from deephyper.evaluator import Evaluator

        with pytest.raises(ValueError):
            evaluator = Evaluator.create(
                run,
                method="threadPool",
                method_kwargs={
                    "num_workers": 1,
                },
            )

    def execute_evaluator(self, method):
        from deephyper.evaluator import Evaluator

        # without kwargs
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
        for config, job in zip(configs, jobs):
            assert config["x"] == job.config["x"]
            assert config["x"] == job.result

        evaluator.submit(configs)
        jobs = evaluator.gather("BATCH", size=1)
        assert 1 <= len(jobs) and len(jobs) <= len(configs)

        # with kwargs
        evaluator = Evaluator.create(
            run,
            method=method,
            method_kwargs={"num_workers": 1, "run_function_kwargs": {"y": 1}},
        )

        configs = [{"x": i} for i in range(10)]
        evaluator.submit(configs)
        jobs = evaluator.gather("ALL")
        jobs.sort(key=lambda j: j.config["x"])
        for config, job in zip(configs, jobs):
            assert config["x"] == job.config["x"]
            assert job.result == config["x"] + 1

        evaluator.submit(configs)
        jobs = evaluator.gather("BATCH", size=1)
        assert 1 <= len(jobs) and len(jobs) <= len(configs)

        # many results
        evaluator = Evaluator.create(
            run_many_results,
            method=method,
            method_kwargs={
                "num_workers": 1,
            },
        )

        configs = [{"x": i} for i in range(10)]
        evaluator.submit(configs)
        jobs = evaluator.gather("ALL")
        jobs.sort(key=lambda j: j.config["x"])
        for config, job in zip(configs, jobs):
            assert config["x"] == job.config["x"]
            assert type(job.result) is dict
            assert job.result["x"] == config["x"]
            assert job.result["y"] == 0

    @pytest.mark.fast
    def test_serial(self):
        self.execute_evaluator("serial")

    @pytest.mark.fast
    def test_thread(self):
        self.execute_evaluator("thread")

    @pytest.mark.fast
    def test_process(self):
        self.execute_evaluator("process")

    @pytest.mark.fast
    def test_subprocess(self):
        self.execute_evaluator("subprocess")

    @pytest.mark.ray
    @pytest.mark.slow
    def test_ray(self):
        try:
            self.execute_evaluator("ray")
        except ModuleNotFoundError as e:
            e_str = str(e)
            if not ("ray" in e_str):
                raise e

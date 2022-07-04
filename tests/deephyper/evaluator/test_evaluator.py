import unittest
from collections import Counter

import pandas as pd
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

    @pytest.mark.fast
    def test_run_function_standards(self):
        from deephyper.evaluator import SerialEvaluator

        configs = [{"x": i} for i in range(10)]

        # float for single objective optimization
        def run(config):
            return 42.0

        evaluator = SerialEvaluator(run)
        evaluator.submit(configs)
        evaluator.gather(type="ALL")
        evaluator.dump_evals()
        results = pd.read_csv("results.csv")
        assert all(
            results.columns
            == ["x", "job_id", "objective", "timestamp_submit", "timestamp_gather"]
        )
        assert len(results) == 10
        assert results["objective"][0] == 42.0

        # str with "F" prefix for failed evaluation
        def run(config):
            return "F_out_of_memory"

        evaluator = SerialEvaluator(run)
        evaluator.submit(configs)
        evaluator.gather(type="ALL")
        evaluator.dump_evals()
        results = pd.read_csv("results.csv")
        assert all(
            results.columns
            == ["x", "job_id", "objective", "timestamp_submit", "timestamp_gather"]
        )
        assert len(results) == 10
        assert results["objective"][0] == "F_out_of_memory"

        # dict
        def run(config):
            return {"objective": 42.0}

        evaluator = SerialEvaluator(run)
        evaluator.submit(configs)
        evaluator.gather(type="ALL")
        evaluator.dump_evals()
        results = pd.read_csv("results.csv")
        assert all(
            results.columns
            == ["x", "job_id", "objective", "timestamp_submit", "timestamp_gather"]
        )
        assert len(results) == 10
        assert results["objective"][0] == 42.0

        # dict with additional information
        def run(config):
            return {
                "objective": 42.0,
                "num_epochs_trained": 25,
                "num_parameters": 420000,
            }

        evaluator = SerialEvaluator(run)
        evaluator.submit(configs)
        evaluator.gather(type="ALL")
        evaluator.dump_evals()
        results = pd.read_csv("results.csv")
        assert all(
            results.columns
            == [
                "x",
                "job_id",
                "objective",
                "num_epochs_trained",
                "num_parameters",
                "timestamp_submit",
                "timestamp_gather",
            ]
        )
        assert len(results) == 10
        assert results["objective"][0] == 42.0
        assert results["num_epochs_trained"][0] == 25
        assert results["num_parameters"][0] == 420000

        # dict with reserved keywords (when @profile decorator is used)
        from deephyper.evaluator import profile

        @profile
        def run(config):
            return 42.0

        evaluator = SerialEvaluator(run)
        evaluator.submit(configs)
        evaluator.gather(type="ALL")
        evaluator.dump_evals()
        results = pd.read_csv("results.csv")
        assert all(
            results.columns
            == [
                "x",
                "job_id",
                "objective",
                "timestamp_submit",
                "timestamp_gather",
                "timestamp_start",
                "timestamp_end",
            ]
        )
        assert len(results) == 10
        assert results["objective"][0] == 42.0

        # combine previous the two previous tests
        @profile
        def run(config):
            return {
                "objective": 42.0,
                "num_epochs_trained": 25,
                "num_parameters": 420000,
            }

        evaluator = SerialEvaluator(run)
        evaluator.submit(configs)
        evaluator.gather(type="ALL")
        evaluator.dump_evals()
        results = pd.read_csv("results.csv")
        assert all(
            results.columns
            == [
                "x",
                "job_id",
                "objective",
                "num_epochs_trained",
                "num_parameters",
                "timestamp_submit",
                "timestamp_gather",
                "timestamp_start",
                "timestamp_end",
            ]
        )
        assert len(results) == 10
        assert results["objective"][0] == 42.0
        assert results["num_epochs_trained"][0] == 25
        assert results["num_parameters"][0] == 420000

        # tuple of float for multi-objective optimization (will appear as "objective_0" and "objective_1" in the resulting dataframe)
        def run(config):
            if config["x"] < 5:
                return 42.0, 0.42
            else:
                return "F_out_of_memory"

        evaluator = SerialEvaluator(run)
        evaluator.num_objective = 2
        evaluator.submit(configs)
        evaluator.gather(type="ALL")
        evaluator.dump_evals()
        results = pd.read_csv("results.csv")
        assert all(
            results.columns
            == [
                "x",
                "job_id",
                "objective_0",
                "objective_1",
                "timestamp_submit",
                "timestamp_gather",
            ]
        )
        assert len(results) == 10

        counter = Counter(results["objective_0"])
        assert counter["42.0"] == 5 and counter["F_out_of_memory"] == 5

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


if __name__ == "__main__":
    test = TestEvaluator()
    test.test_run_function_standards()

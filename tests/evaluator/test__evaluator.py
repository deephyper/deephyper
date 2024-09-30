import unittest
from collections import Counter

import pandas as pd
import pytest


def run(job, y=0):
    return job["x"] + y


def run_many_results(job, y=0):
    return {"objective": job["x"], "metadata": {"y": y}}


class TestEvaluatorWithHPOJob(unittest.TestCase):
    @pytest.mark.fast
    @pytest.mark.hps
    def test_import(self):
        from deephyper.evaluator import Evaluator

    @pytest.mark.fast
    @pytest.mark.hps
    def test_wrong_evaluator(self):
        from deephyper.evaluator import Evaluator

        with pytest.raises(ValueError):
            evaluator = Evaluator.create(
                run,
                method="threads",
                method_kwargs={
                    "num_workers": 1,
                },
            )

    @pytest.mark.fast
    @pytest.mark.hps
    def test_run_function_standards(self):
        from deephyper.evaluator import SerialEvaluator, HPOJob

        configs = [{"x": i} for i in range(10)]

        # float for single objective optimization
        def run(config):
            return 42.0

        evaluator = SerialEvaluator(run)
        evaluator._job_class = HPOJob

        evaluator.submit(configs)
        evaluator.gather(type="ALL")
        evaluator.dump_jobs_done_to_csv()
        results = pd.read_csv("results.csv")
        assert all(
            results.columns
            == [
                "p:x",
                "objective",
                "job_id",
                "m:timestamp_submit",
                "m:timestamp_gather",
            ]
        )
        assert len(results) == 10
        assert results["objective"][0] == 42.0

        # str with "F" prefix for failed evaluation
        def run(config):
            if config["x"] < 5:
                return "F_out_of_memory"
            else:
                return 42.0

        evaluator = SerialEvaluator(run)
        evaluator._job_class = HPOJob
        evaluator.submit(configs)
        evaluator.gather(type="ALL")
        evaluator.dump_jobs_done_to_csv()
        results = pd.read_csv("results.csv").sort_values(by="job_id")
        assert all(
            results.columns
            == [
                "p:x",
                "objective",
                "job_id",
                "m:timestamp_submit",
                "m:timestamp_gather",
            ]
        )
        assert len(results) == 10
        assert results.iloc[0]["objective"] == "F_out_of_memory"

        # dict
        def run(config):
            return {"objective": 42.0}

        evaluator = SerialEvaluator(run)
        evaluator._job_class = HPOJob
        evaluator.submit(configs)
        evaluator.gather(type="ALL")
        evaluator.dump_jobs_done_to_csv()
        results = pd.read_csv("results.csv")
        assert all(
            results.columns
            == [
                "p:x",
                "objective",
                "job_id",
                "m:timestamp_submit",
                "m:timestamp_gather",
            ]
        )
        assert len(results) == 10
        assert results["objective"][0] == 42.0

        # dict with additional information
        def run(config):
            return {
                "objective": 42.0,
                "metadata": {"num_epochs_trained": 25, "num_parameters": 420000},
            }

        evaluator = SerialEvaluator(run)
        evaluator._job_class = HPOJob
        evaluator.submit(configs)
        evaluator.gather(type="ALL")
        evaluator.dump_jobs_done_to_csv()
        results = pd.read_csv("results.csv")
        assert list(sorted(results.columns)) == list(
            sorted(
                [
                    "p:x",
                    "objective",
                    "job_id",
                    "m:timestamp_submit",
                    "m:timestamp_gather",
                    "m:num_epochs_trained",
                    "m:num_parameters",
                ]
            )
        )

        assert len(results) == 10
        assert results["objective"][0] == 42.0
        assert results["m:num_epochs_trained"][0] == 25
        assert results["m:num_parameters"][0] == 420000

        # dict with reserved keywords (when @profile decorator is used)
        from deephyper.evaluator import profile

        @profile
        def run(config):
            return 42.0

        evaluator = SerialEvaluator(run)
        evaluator._job_class = HPOJob
        evaluator.submit(configs)
        evaluator.gather(type="ALL")
        evaluator.dump_jobs_done_to_csv()
        results = pd.read_csv("results.csv")
        assert list(sorted(results.columns)) == list(
            sorted(
                [
                    "p:x",
                    "objective",
                    "job_id",
                    "m:timestamp_submit",
                    "m:timestamp_gather",
                    "m:timestamp_start",
                    "m:timestamp_end",
                ]
            )
        )
        assert len(results) == 10
        assert results["objective"][0] == 42.0

        # combine previous the two previous tests
        @profile
        def run(config):
            return {
                "objective": 42.0,
                "metadata": {
                    "num_epochs_trained": 25,
                    "num_parameters": 420000,
                },
            }

        evaluator = SerialEvaluator(run)
        evaluator._job_class = HPOJob
        evaluator.submit(configs)
        evaluator.gather(type="ALL")
        evaluator.dump_jobs_done_to_csv()
        results = pd.read_csv("results.csv")
        assert list(sorted(results.columns)) == list(
            sorted(
                [
                    "p:x",
                    "objective",
                    "job_id",
                    "m:timestamp_submit",
                    "m:timestamp_gather",
                    "m:timestamp_start",
                    "m:timestamp_end",
                    "m:num_epochs_trained",
                    "m:num_parameters",
                ]
            )
        )
        assert len(results) == 10
        assert results["objective"][0] == 42.0
        assert results["m:num_epochs_trained"][0] == 25
        assert results["m:num_parameters"][0] == 420000

        # tuple of float for multi-objective optimization (will appear as "objective_0" and "objective_1" in the resulting dataframe)
        def run(config):
            if config["x"] < 5:
                return 42.0, 0.42
            else:
                return "F_out_of_memory"

        evaluator = SerialEvaluator(run)
        evaluator._job_class = HPOJob
        evaluator.num_objective = 2
        evaluator.submit(configs)
        evaluator.gather(type="ALL")
        evaluator.dump_jobs_done_to_csv()
        results = pd.read_csv("results.csv")
        assert list(sorted(results.columns)) == list(
            sorted(
                [
                    "p:x",
                    "objective_0",
                    "objective_1",
                    "job_id",
                    "m:timestamp_submit",
                    "m:timestamp_gather",
                ]
            )
        )
        assert len(results) == 10

        counter = Counter(results["objective_0"])
        assert counter["42.0"] == 5 and counter["F_out_of_memory"] == 5

    def execute_evaluator(self, method):
        from deephyper.evaluator import Evaluator, HPOJob

        # without kwargs
        method_kwargs = {"num_workers": 1}
        if method == "ray":
            import os

            HERE = os.path.dirname(os.path.abspath(__file__))
            method_kwargs["ray_kwargs"] = {"runtime_env": {"working_dir": HERE}}

        evaluator = Evaluator.create(run, method=method, method_kwargs=method_kwargs)
        evaluator._job_class = HPOJob

        configs = [{"x": i} for i in range(10)]
        evaluator.submit(configs)
        jobs = evaluator.gather("ALL")
        jobs.sort(key=lambda j: j.args["x"])
        for config, job in zip(configs, jobs):
            assert config["x"] == job.args["x"]
            assert config["x"] == job.objective
        evaluator.submit(configs)
        jobs = evaluator.gather("BATCH", size=1)
        assert 1 <= len(jobs) and len(jobs) <= len(configs)

        # with kwargs
        evaluator = Evaluator.create(
            run,
            method=method,
            method_kwargs={"num_workers": 1, "run_function_kwargs": {"y": 1}},
        )
        evaluator._job_class = HPOJob

        configs = [{"x": i} for i in range(10)]
        evaluator.submit(configs)
        jobs = evaluator.gather("ALL")
        jobs.sort(key=lambda j: j.args["x"])
        for config, job in zip(configs, jobs):
            assert config["x"] == job.args["x"]
            assert job.objective == config["x"] + 1

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
        evaluator._job_class = HPOJob

        configs = [{"x": i} for i in range(10)]
        evaluator.submit(configs)
        jobs = evaluator.gather("ALL")
        jobs.sort(key=lambda j: j.args["x"])
        for config, job in zip(configs, jobs):
            assert config["x"] == job.args["x"]
            assert type(job.output) is dict
            assert job.objective == config["x"]
            assert job.metadata["y"] == 0

    @pytest.mark.fast
    @pytest.mark.hps
    def test_serial(self):
        self.execute_evaluator("serial")

    @pytest.mark.fast
    @pytest.mark.hps
    def test_thread(self):
        self.execute_evaluator("thread")

    @pytest.mark.fast
    @pytest.mark.hps
    def test_process(self):
        self.execute_evaluator("process")

    @pytest.mark.fast
    @pytest.mark.hps
    @pytest.mark.ray
    def test_ray(self):
        try:
            self.execute_evaluator("ray")
        except ModuleNotFoundError as e:
            e_str = str(e)
            if not ("ray" in e_str):
                raise e


def run_job_scalar_output(job):
    return 0


def run_job_dict_output(job):
    x1 = job.parameters["x1"]
    x2 = job.parameters["x2"]
    return {"y1": x1, "y2": x2, "list": [x1, x2], "dict": {"x1": x1, "x2": x2}}


def test_evaluator_with_Job(tmp_path):
    import os
    import pandas as pd
    from deephyper.evaluator import Evaluator

    # Scalar output
    evaluator = Evaluator.create(run_job_scalar_output, method="serial")
    input_parameters = [{"x": i} for i in range(10)]
    evaluator.submit(input_parameters)
    jobs_done = evaluator.gather("ALL")
    for jobi in jobs_done:
        assert jobi.output == 0
    evaluator.dump_jobs_done_to_csv(log_dir=tmp_path)
    df = pd.read_csv(os.path.join(tmp_path, "results.csv"))
    assert len(df) == 10
    assert len(df.columns) == 5

    # Dict output
    evaluator = Evaluator.create(run_job_dict_output, method="serial")
    input_parameters = [{"x1": i, "x2": i + 1} for i in range(10)]
    evaluator.submit(input_parameters)
    jobs_done = evaluator.gather("ALL")
    for jobi in jobs_done:
        assert isinstance(jobi.output, dict)
    evaluator.dump_jobs_done_to_csv(log_dir=tmp_path)
    df = pd.read_csv(os.path.join(tmp_path, "results.csv"))
    assert len(df) == 10
    assert len(df.columns) == 9


if __name__ == "__main__":
    tmp_path = "/tmp/deephyper/"
    # test = TestEvaluator()
    # test.test_run_function_standards()
    # test.test_wrong_evaluator()
    test_evaluator_with_Job(".")

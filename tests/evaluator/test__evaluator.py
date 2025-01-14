import asyncio
import os
import pathlib
import time
from collections import Counter

import pandas as pd
import pytest

from deephyper.evaluator import profile


def run_sync(job, y=0):
    # The following line impacts the result as it puts on old a running threading.Thread
    time.sleep(0.01)
    return job["x"] + y


async def run_async(job, y=0):
    # The following line impacts the result as it puts on old a running asyncio.Task
    await asyncio.sleep(0.01)
    return job["x"] + y


def run_many_results_sync(job, y=0):
    # The following line impacts the result as it puts on old a running threading.Thread
    time.sleep(0.01)
    return {"objective": job["x"], "metadata": {"y": y}}


async def run_many_results_async(job, y=0):
    # The following line impacts the result as it puts on old a running asyncio.Task
    await asyncio.sleep(0.01)
    return {"objective": job["x"], "metadata": {"y": y}}


def test_wrong_evaluator():
    from deephyper.evaluator import Evaluator

    with pytest.raises(ValueError):
        Evaluator.create(
            run_sync,
            method="threads",
            method_kwargs={
                "num_workers": 1,
            },
        )


def test_run_function_standards(tmp_path):
    from deephyper.evaluator import HPOJob, SerialEvaluator, ThreadPoolEvaluator

    configs = [{"x": i} for i in range(10)]

    # float for single objective optimization
    def run(config):
        return 42.0

    with pytest.raises(ValueError):
        evaluator = SerialEvaluator(run)

    async def run(job):
        return 42.0

    evaluator = SerialEvaluator(run)
    evaluator._job_class = HPOJob

    evaluator.submit(configs)
    evaluator.gather(type="ALL")
    evaluator.dump_jobs_done_to_csv(log_dir=tmp_path)
    results = pd.read_csv(os.path.join(tmp_path, "results.csv"))
    assert all(
        results.columns
        == [
            "p:x",
            "objective",
            "job_id",
            "job_status",
            "m:timestamp_submit",
            "m:timestamp_gather",
        ]
    )
    assert len(results) == 10
    assert results["objective"][0] == 42.0
    evaluator.close()

    # str with "F" prefix for failed evaluation
    async def run(job):
        if job["x"] < 5:
            return "F_out_of_memory"
        else:
            return 42.0

    evaluator = SerialEvaluator(run)
    evaluator._job_class = HPOJob
    evaluator.submit(configs)
    evaluator.gather(type="ALL")
    evaluator.dump_jobs_done_to_csv(log_dir=tmp_path)
    results = pd.read_csv(os.path.join(tmp_path, "results.csv")).sort_values(by="job_id")
    assert all(
        results.columns
        == [
            "p:x",
            "objective",
            "job_id",
            "job_status",
            "m:timestamp_submit",
            "m:timestamp_gather",
        ]
    )
    assert len(results) == 10
    assert results.iloc[0]["objective"] == "F_out_of_memory"
    evaluator.close()

    # dict
    async def run(job):
        return {"objective": 42.0}

    evaluator = SerialEvaluator(run)
    evaluator._job_class = HPOJob
    evaluator.submit(configs)
    evaluator.gather(type="ALL")
    evaluator.dump_jobs_done_to_csv(log_dir=tmp_path)
    results = pd.read_csv(os.path.join(tmp_path, "results.csv"))
    assert all(
        results.columns
        == [
            "p:x",
            "objective",
            "job_id",
            "job_status",
            "m:timestamp_submit",
            "m:timestamp_gather",
        ]
    )
    assert len(results) == 10
    assert results["objective"][0] == 42.0
    evaluator.close()

    # dict with additional information
    async def run(job):
        return {
            "objective": 42.0,
            "metadata": {"num_epochs_trained": 25, "num_parameters": 420000},
        }

    evaluator = SerialEvaluator(run)
    evaluator._job_class = HPOJob
    evaluator.submit(configs)
    evaluator.gather(type="ALL")
    evaluator.dump_jobs_done_to_csv(log_dir=tmp_path)
    results = pd.read_csv(os.path.join(tmp_path, "results.csv"))
    assert list(sorted(results.columns)) == list(
        sorted(
            [
                "p:x",
                "objective",
                "job_id",
                "job_status",
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
    evaluator.close()

    # dict with reserved keywords (when @profile decorator is used)
    from deephyper.evaluator import profile

    @profile
    async def run(job):
        return 42.0

    evaluator = SerialEvaluator(run)
    evaluator._job_class = HPOJob
    evaluator.submit(configs)
    evaluator.gather(type="ALL")
    evaluator.dump_jobs_done_to_csv(log_dir=tmp_path)
    results = pd.read_csv(os.path.join(tmp_path, "results.csv"))
    assert list(sorted(results.columns)) == list(
        sorted(
            [
                "p:x",
                "objective",
                "job_id",
                "job_status",
                "m:timestamp_submit",
                "m:timestamp_gather",
                "m:timestamp_start",
                "m:timestamp_end",
            ]
        )
    )
    assert len(results) == 10
    assert results["objective"][0] == 42.0
    evaluator.close()

    # combine the two previous tests
    @profile
    async def run(job):
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
    evaluator.dump_jobs_done_to_csv(log_dir=tmp_path)
    results = pd.read_csv(os.path.join(tmp_path, "results.csv"))
    assert list(sorted(results.columns)) == list(
        sorted(
            [
                "p:x",
                "objective",
                "job_id",
                "job_status",
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
    evaluator.close()

    # do the previous test with sync function and thread evaluator
    @profile
    def run(job):
        return {
            "objective": 42.0,
            "metadata": {
                "num_epochs_trained": 25,
                "num_parameters": 420000,
            },
        }

    evaluator = ThreadPoolEvaluator(run)
    evaluator._job_class = HPOJob
    evaluator.submit(configs)
    evaluator.gather(type="ALL")
    evaluator.dump_jobs_done_to_csv(log_dir=tmp_path)
    results = pd.read_csv(os.path.join(tmp_path, "results.csv"))
    assert list(sorted(results.columns)) == list(
        sorted(
            [
                "p:x",
                "objective",
                "job_id",
                "job_status",
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
    evaluator.close()

    # Tuple of float for multi-objective optimization
    # It will appear as "objective_0" and "objective_1" in the results
    async def run(config):
        if config["x"] < 5:
            return 42.0, 0.42
        else:
            return "F_out_of_memory"

    evaluator = SerialEvaluator(run)
    evaluator._job_class = HPOJob
    evaluator.num_objective = 2
    evaluator.submit(configs)
    evaluator.gather(type="ALL")
    evaluator.dump_jobs_done_to_csv(log_dir=tmp_path)
    results = pd.read_csv(os.path.join(tmp_path, "results.csv"))
    assert list(sorted(results.columns)) == list(
        sorted(
            [
                "p:x",
                "objective_0",
                "objective_1",
                "job_id",
                "job_status",
                "m:timestamp_submit",
                "m:timestamp_gather",
            ]
        )
    )
    assert len(results) == 10
    evaluator.close()

    counter = Counter(results["objective_0"])
    assert counter["42.0"] == 5 and counter["F_out_of_memory"] == 5


def execute_evaluator(method, tmp_path):
    from deephyper.evaluator import Evaluator, HPOJob

    # Create the directory where results from tests are potentially saved
    pathlib.Path(tmp_path).mkdir(parents=True, exist_ok=True)

    # without kwargs
    method_kwargs = {"num_workers": 1}
    if method == "ray":
        HERE = os.path.dirname(os.path.abspath(__file__))
        method_kwargs["ray_kwargs"] = {"runtime_env": {"working_dir": HERE}}

    evaluator = Evaluator.create(
        run_async if method == "serial" else run_sync,
        method=method,
        method_kwargs=method_kwargs,
    )
    evaluator._job_class = HPOJob

    configs = [{"x": i} for i in range(10)]
    evaluator.submit(configs)
    jobs = evaluator.gather("ALL")
    jobs.sort(key=lambda j: j.args["x"])
    assert len(jobs) == 10
    for config, job in zip(configs, jobs):
        assert config["x"] == job.args["x"]
        assert config["x"] == job.objective
    evaluator.submit(configs)
    jobs = evaluator.gather("BATCH", size=1)
    assert len(jobs) == 1
    evaluator.close()

    evaluator.dump_jobs_done_to_csv(log_dir=tmp_path)
    result = pd.read_csv(os.path.join(tmp_path, "results.csv"))
    assert len(result) == 20
    assert all(s == "CANCELLED" for s in result["job_status"].iloc[-9:])

    # with kwargs
    evaluator = Evaluator.create(
        run_async if method == "serial" else run_sync,
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
    evaluator.close()

    # many results
    evaluator = Evaluator.create(
        run_many_results_async if method == "serial" else run_many_results_sync,
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
    evaluator.close()


def test_serial(tmp_path):
    execute_evaluator("serial", tmp_path)


def test_thread(tmp_path):
    execute_evaluator("thread", tmp_path)


def test_process(tmp_path):
    execute_evaluator("process", tmp_path)


@pytest.mark.ray
def test_ray(tmp_path):
    try:
        execute_evaluator("ray", tmp_path)
    except ModuleNotFoundError as e:
        e_str = str(e)
        if "ray" not in e_str:
            raise e


async def run_job_scalar_output_async(job):
    return 0


async def run_job_dict_output_async(job):
    x1 = job.parameters["x1"]
    x2 = job.parameters["x2"]
    return {"y1": x1, "y2": x2, "list": [x1, x2], "dict": {"x1": x1, "x2": x2}}


@profile
async def run_job_scalar_output_profiled_async(job):
    return 0


@profile
async def run_job_dict_output_profiled_async(job):
    x1 = job.parameters["x1"]
    x2 = job.parameters["x2"]
    return {"y1": x1, "y2": x2, "list": [x1, x2], "dict": {"x1": x1, "x2": x2}}


def test_evaluator_with_Job(tmp_path):
    from deephyper.evaluator import Evaluator

    # Scalar output
    evaluator = Evaluator.create(run_job_scalar_output_async, method="serial")
    input_parameters = [{"x": i} for i in range(10)]
    evaluator.submit(input_parameters)
    jobs_done = evaluator.gather("ALL")
    for jobi in jobs_done:
        assert jobi.output == 0
    evaluator.dump_jobs_done_to_csv(log_dir=tmp_path)
    df = pd.read_csv(os.path.join(tmp_path, "results.csv"))
    assert len(df) == 10
    assert len(df.columns) == 6
    evaluator.close()

    # Dict output
    evaluator = Evaluator.create(run_job_dict_output_async, method="serial")
    input_parameters = [{"x1": i, "x2": i + 1} for i in range(10)]
    evaluator.submit(input_parameters)
    jobs_done = evaluator.gather("ALL")
    for jobi in jobs_done:
        assert isinstance(jobi.output, dict)
    evaluator.dump_jobs_done_to_csv(log_dir=tmp_path)
    df = pd.read_csv(os.path.join(tmp_path, "results.csv"))
    assert len(df) == 10
    assert len(df.columns) == 10
    evaluator.close()

    # Scalar output with profiled function
    evaluator = Evaluator.create(run_job_scalar_output_profiled_async, method="serial")
    input_parameters = [{"x": i} for i in range(10)]
    evaluator.submit(input_parameters)
    jobs_done = evaluator.gather("ALL")
    for jobi in jobs_done:
        assert jobi.output == 0
        assert "timestamp_start" in jobi.metadata
        assert "timestamp_end" in jobi.metadata
    evaluator.dump_jobs_done_to_csv(log_dir=tmp_path)
    df = pd.read_csv(os.path.join(tmp_path, "results.csv"))
    assert len(df) == 10
    assert len(df.columns) == 8
    evaluator.close()

    # Dict output with profiled function
    evaluator = Evaluator.create(run_job_dict_output_profiled_async, method="serial")
    input_parameters = [{"x1": i, "x2": i + 1} for i in range(10)]
    evaluator.submit(input_parameters)
    jobs_done = evaluator.gather("ALL")
    for jobi in jobs_done:
        assert isinstance(jobi.output, dict)
        assert jobi.output["y1"] == jobi.args["x1"]
        assert jobi.output["y2"] == jobi.args["x2"]
        assert "timestamp_start" in jobi.metadata
        assert "timestamp_end" in jobi.metadata
    evaluator.dump_jobs_done_to_csv(log_dir=tmp_path)
    df = pd.read_csv(os.path.join(tmp_path, "results.csv"))
    assert len(df) == 10
    assert len(df.columns) == 12
    evaluator.close()


if __name__ == "__main__":
    import logging

    logging.basicConfig(
        # filename="deephyper.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s - %(message)s",
        force=True,
    )

    tmp_path = "/tmp/deephyper_test/"
    # test_run_function_standards(tmp_path)
    # test_serial(tmp_path)
    # test_thread(tmp_path)
    # test_process(tmp_path)
    # test_ray(tmp_path)
    # test_wrong_evaluator()
    test_evaluator_with_Job(tmp_path)

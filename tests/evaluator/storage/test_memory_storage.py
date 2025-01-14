from deephyper.evaluator import Evaluator, RunningJob, HPOJob
from deephyper.evaluator.storage import MemoryStorage


async def run_async(job: RunningJob) -> dict:
    return {
        "objective": job.parameters["x"],
        "metadata": {"storage_id": id(job.storage)},
    }


def run_sync(job: RunningJob) -> dict:
    return {
        "objective": job.parameters["x"],
        "metadata": {"storage_id": id(job.storage)},
    }


def test_basic():
    # Creation of the database
    storage = MemoryStorage()
    search_id0 = storage.create_new_search()
    job_id0 = storage.create_new_job(search_id0)
    job_id1 = storage.create_new_job(search_id=search_id0)

    assert search_id0 == "0"
    assert job_id0 == "0.0"
    assert job_id1 == "0.1"

    search_id1 = storage.create_new_search()
    job_id0 = storage.create_new_job(search_id1)
    job_id1 = storage.create_new_job(search_id=search_id1)

    assert search_id1 == "1"
    assert job_id0 == "1.0"
    assert job_id1 == "1.1"

    # Check available ids
    search_ids = storage.load_all_search_ids()
    assert search_ids == ["0", "1"]

    job_ids = storage.load_all_job_ids(search_id0)
    assert job_ids == ["0.0", "0.1"]

    # Store/Load
    # Job is empty
    job_id0_data = storage.load_job(job_id0)
    assert "in" in job_id0_data
    assert "out" in job_id0_data
    assert "metadata" in job_id0_data

    # Storing inputs of job
    storage.store_job_in(job_id0, args=(1, 2), kwargs={"foo": 0})
    job_id0_data = storage.load_job(job_id0)
    assert "args" in job_id0_data["in"]
    assert "kwargs" in job_id0_data["in"]
    assert job_id0_data["in"]["args"] == (1, 2)
    assert job_id0_data["in"]["kwargs"] == {"foo": 0}

    # Storing outputs of job
    storage.store_job_out(job_id0, 0)
    assert job_id0_data["out"] is None
    job_id0_data = storage.load_job(job_id0)
    assert 0 == job_id0_data["out"]

    # Storing metadata of job
    storage.store_job_metadata(job_id0, "timestamp", 10)
    assert job_id0_data["metadata"] == {}
    job_id0_data = storage.load_job(job_id0)
    assert job_id0_data["metadata"] == {"timestamp": 10}


def test_with_evaluator():
    storage = MemoryStorage()

    # serial evaluator
    evaluator = Evaluator.create(run_async, method="serial", method_kwargs={"storage": storage})
    evaluator._job_class = HPOJob
    evaluator.submit([{"x": 0}])
    job_done = evaluator.gather("ALL")[0]
    assert job_done.metadata["storage_id"] == id(storage)
    evaluator.close()

    # thread evaluator
    evaluator = Evaluator.create(run_sync, method="thread", method_kwargs={"storage": storage})
    evaluator._job_class = HPOJob
    evaluator.submit([{"x": 0}])
    job_done = evaluator.gather("ALL")[0]
    assert job_done.metadata["storage_id"] == id(storage)
    evaluator.close()

    # process evaluator
    evaluator = Evaluator.create(run_sync, method="process", method_kwargs={"storage": storage})
    evaluator._job_class = HPOJob
    evaluator.submit([{"x": 0}])
    job_done = evaluator.gather("ALL")[0]
    assert job_done.metadata["storage_id"] != id(storage)
    evaluator.close()

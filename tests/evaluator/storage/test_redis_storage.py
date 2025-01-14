import pytest
from deephyper.evaluator import Evaluator, RunningJob, HPOJob


async def run_0_async(job: RunningJob) -> dict:
    job.storage.store_job_metadata(job.id, "foo", 0)
    return {
        "objective": job.parameters["x"],
        "metadata": {"storage_id": id(job.storage)},
    }


def run_0_sync(job: RunningJob) -> dict:
    job.storage.store_job_metadata(job.id, "foo", 0)
    return {
        "objective": job.parameters["x"],
        "metadata": {"storage_id": id(job.storage)},
    }


@pytest.mark.redis
def test_basic():
    from deephyper.evaluator.storage._redis_storage import RedisStorage

    # Creation of the database
    storage = RedisStorage()
    storage.connect()
    storage._redis.flushdb()  # empty the db before using it
    search_id0 = storage.create_new_search()
    job_id0 = storage.create_new_job(search_id0)
    job_id1 = storage.create_new_job(search_id=search_id0)

    assert search_id0 == "0"
    assert job_id0 == "0.0"
    assert job_id1 == "0.1"

    job_ids = storage.load_all_job_ids(search_id0)
    assert len(job_ids) == 2

    # Check outputs of jobs
    job_outputs = storage.load_out_from_all_jobs(search_id0)
    assert len(job_outputs) == 0

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
    assert job_id0_data["in"]["args"] == [1, 2]
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

    # Storing metadata of job with a NaN value
    storage.store_job_metadata(job_id0, "nan_value", float("nan"))
    job_id0_data = storage.load_job(job_id0)
    assert job_id0_data["metadata"]["nan_value"] == "NaN"


@pytest.mark.redis
def test_with_evaluator():
    from deephyper.evaluator.storage._redis_storage import RedisStorage

    storage = RedisStorage()
    storage.connect()
    storage._redis.flushdb()

    # serial evaluator
    evaluator = Evaluator.create(run_0_async, method="serial", method_kwargs={"storage": storage})
    evaluator._job_class = HPOJob
    evaluator.submit([{"x": 0}])
    job_done = evaluator.gather("ALL")[0]
    assert job_done.metadata["storage_id"] == id(storage)

    # thread evaluator
    evaluator = Evaluator.create(run_0_sync, method="thread", method_kwargs={"storage": storage})
    evaluator._job_class = HPOJob
    evaluator.submit([{"x": 0}])
    job_done = evaluator.gather("ALL")[0]
    assert job_done.metadata["storage_id"] == id(storage)

    # process evaluator
    evaluator = Evaluator.create(run_0_sync, method="process", method_kwargs={"storage": storage})
    evaluator._job_class = HPOJob
    evaluator.submit([{"x": 0}])
    job_done = evaluator.gather("ALL")[0]
    assert job_done.metadata["storage_id"] != id(storage)

    data = evaluator._storage.load_search(evaluator._search_id)
    assert data[0]["metadata"]["foo"] == 0

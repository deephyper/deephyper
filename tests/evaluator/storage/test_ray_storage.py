import math
import unittest
import pytest

from deephyper.evaluator import Evaluator, RunningJob


async def run_async(job: RunningJob) -> dict:
    if not job.storage.is_connected():
        job.storage.connect()
    job.storage.store_job_metadata(job.id, "foo", 0)
    return {
        "objective": job.parameters["x"],
        "metadata": {"storage_id": id(job.storage)},
    }


def run_sync(job: RunningJob) -> dict:
    if not job.storage.is_connected():
        job.storage.connect()
    job.storage.store_job_metadata(job.id, "foo", 0)
    return {
        "objective": job.parameters["x"],
        "metadata": {"storage_id": id(job.storage)},
    }


@pytest.mark.fast
@pytest.mark.ray
class TestRayStorage(unittest.TestCase):
    def test_basic(self):
        import ray
        from deephyper.evaluator.storage._ray_storage import RayStorage

        # Init Ray
        if not (ray.is_initialized()):
            ray.init()

        # Creation of the database
        storage = RayStorage()
        storage.connect()

        search_id0 = storage.create_new_search()
        job_id0 = storage.create_new_job(search_id0)
        job_id1 = storage.create_new_job(search_id=search_id0)

        self.assertEqual(search_id0, "0")
        self.assertEqual(job_id0, "0.0")
        self.assertEqual(job_id1, "0.1")

        search_id1 = storage.create_new_search()
        job_id0 = storage.create_new_job(search_id1)
        job_id1 = storage.create_new_job(search_id=search_id1)

        self.assertEqual(search_id1, "1")
        self.assertEqual(job_id0, "1.0")
        self.assertEqual(job_id1, "1.1")

        # Check available ids
        search_ids = storage.load_all_search_ids()
        self.assertEqual(search_ids, ["0", "1"])

        job_ids = storage.load_all_job_ids(search_id0)
        self.assertEqual(job_ids, ["0.0", "0.1"])

        # Store/Load
        # Job is empty
        job_id0_data = storage.load_job(job_id0)
        self.assertIn("in", job_id0_data)
        self.assertIn("out", job_id0_data)
        self.assertIn("metadata", job_id0_data)

        # Storing inputs of job
        storage.store_job_in(job_id0, args=(1, 2), kwargs={"foo": 0})
        job_id0_data = storage.load_job(job_id0)
        self.assertIn("args", job_id0_data["in"])
        self.assertIn("kwargs", job_id0_data["in"])
        self.assertEqual(job_id0_data["in"]["args"], (1, 2))
        self.assertEqual(job_id0_data["in"]["kwargs"], {"foo": 0})

        # Storing outputs of job
        storage.store_job_out(job_id0, 0)
        self.assertIs(None, job_id0_data["out"])
        job_id0_data = storage.load_job(job_id0)
        self.assertEqual(0, job_id0_data["out"])

        # Storing metadata of job
        storage.store_job_metadata(job_id0, "timestamp", 10)
        self.assertEqual(job_id0_data["metadata"], {})
        job_id0_data = storage.load_job(job_id0)
        self.assertEqual(job_id0_data["metadata"], {"timestamp": 10})

        # Storing metadata of job with a NaN value
        storage.store_job_metadata(job_id0, "nan_value", float("nan"))
        job_id0_data = storage.load_job(job_id0)
        self.assertTrue(math.isnan(job_id0_data["metadata"]["nan_value"]))

    def test_with_evaluator(self):
        import ray
        from deephyper.evaluator.storage._ray_storage import RayStorage

        # Init Ray
        if not (ray.is_initialized()):
            ray.init()

        # Creation of the database
        storage = RayStorage()
        storage.connect()

        # serial evaluator
        print("serial evaluator")
        evaluator = Evaluator.create(
            run_async, method="serial", method_kwargs={"storage": storage}
        )
        evaluator.submit([{"x": 0}])
        job_done = evaluator.gather("ALL")[0]
        assert job_done.metadata["storage_id"] == id(storage)
        evaluator.close()

        # thread evaluator
        print("thread evaluator")
        evaluator = Evaluator.create(
            run_sync, method="thread", method_kwargs={"storage": storage}
        )
        evaluator.submit([{"x": 0}])
        job_done = evaluator.gather("ALL")[0]
        assert job_done.metadata["storage_id"] == id(storage)
        evaluator.close()

        # process evaluator
        print("process evaluator")
        evaluator = Evaluator.create(
            run_sync, method="process", method_kwargs={"storage": storage}
        )
        evaluator.submit([{"x": 0}])
        job_done = evaluator.gather("ALL")[0]
        assert job_done.metadata["storage_id"] != id(storage)
        evaluator.close()

        data = evaluator._storage.load_search(evaluator._search_id)
        assert data["0"]["metadata"]["foo"] == 0

        # TODO: manage possible warnings and leaked ressources.


if __name__ == "__main__":
    test = TestRayStorage()
    test.test_with_evaluator()

import os
import sys
import time
import pytest

from deephyper.evaluator import Evaluator, HPOJob, RunningJob

PYTHON = sys.executable
SCRIPT = os.path.abspath(__file__)

import deephyper.tests as dht  # noqa: E402


def _test_mpi_win_mutable_mapping():
    from deephyper.evaluator.mpi import MPI
    from deephyper.evaluator.storage._mpi_win_mutable_mapping import (
        MPIWinMutableMapping,
    )

    if not MPI.Is_initialized():
        MPI.Init_thread()

    comm = MPI.COMM_WORLD
    mapping = MPIWinMutableMapping(comm=comm, size=1024 * 100, root=1)

    if comm.Get_rank() > -1:
        time.sleep(0.01 * comm.Get_rank())
        mapping[f"key_{comm.Get_rank()}"] = comm.Get_rank()

    for i in range(comm.Get_size()):
        if i == comm.Get_rank():
            print(f"Process {comm.Get_rank()} has: {mapping} with {len(mapping)} elements")
            assert len(mapping) == i + 1

    comm.Barrier()

    # Test incr regular key
    if comm.Get_rank() == 0:
        mapping["counter"] = 0
        value = mapping.incr("counter")
        assert value == 1
    comm.Barrier()

    value = mapping["counter"]
    assert value == 1
    comm.Barrier()
    counter = mapping.incr("counter")
    comm.Barrier()
    counter = mapping["counter"]
    print(f"Process {comm.Get_rank()} has: {counter=}")
    assert counter == (1 + comm.Get_size())
    comm.Barrier()

    # Test incr with JSON path key
    if comm.Get_rank() == 0:
        mapping["counters"] = {f"counter_{i}": 0 for i in range(comm.Get_size())}
    comm.Barrier()
    counters = mapping["counters"]
    print(f"Process {comm.Get_rank()} has: {counters=}")
    comm.Barrier()

    value = mapping.incr(f"counters.counter_{comm.Get_rank()}")
    comm.Barrier()

    counters = mapping["counters"]
    print(f"Process {comm.Get_rank()} has: {counters=}")
    assert len(counters) == comm.Get_size()
    assert all(i == 1 for i in counters.values())
    comm.Barrier()

    # Delete and create new mapping before next test
    del mapping

    class CustomMPIWinMutableMapping(MPIWinMutableMapping):
        def __init__(self, *args, **kwargs):
            self.count_read = 0
            self.count_write = 0
            super().__init__(*args, **kwargs)
            self.count_write = 0
            self.count_write = 0

        def _read_dict(self):
            self.count_read += 1
            super()._read_dict()

        def _write_dict(self):
            self.count_write += 1
            super()._write_dict()

    mapping = CustomMPIWinMutableMapping(
        default_value={"key_0": 0, "key_1": 1},
        comm=comm,
        size=1024 * 100,
        root=0,
    )

    # Test session
    if comm.Get_rank() == 0:
        mapping.count_read = 0
        mapping.count_write = 0

        # No session was used here
        assert mapping._session_is_started is False
        for key in mapping:
            mapping[key]

        assert mapping.count_read == 5

    comm.Barrier()

    if comm.Get_rank() == 0:
        mapping.count_read = 0
        mapping.count_write = 0

        # Read-only session
        with mapping(ready_only=True):
            assert mapping._session_is_started is True

            for key in mapping:
                mapping[key]

        assert mapping.count_read == 1

    comm.Barrier()


@pytest.mark.mpi
def test_mpi_win_mutable_mapping():
    command = f"mpirun -np 4 {PYTHON} {SCRIPT} _test_mpi_win_mutable_mapping"
    result = dht.run(command, live_output=False)
    assert result.returncode == 0


def _test_mpi_win_storage_basic():
    from deephyper.evaluator.mpi import MPI
    from deephyper.evaluator.storage._mpi_win_storage import MPIWinStorage

    if not MPI.Is_initialized():
        MPI.Init_thread()

    comm = MPI.COMM_WORLD

    # Creation of the database
    storage = MPIWinStorage(comm)
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


@pytest.mark.mpi
def test_mpi_win_storage_basic():
    command = f"mpirun -np 1 {PYTHON} {SCRIPT} _test_mpi_win_storage_basic"
    result = dht.run(command, live_output=False)
    assert result.returncode == 0


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


def _test_mpi_win_storage_with_evaluator():
    from deephyper.evaluator.mpi import MPI
    from deephyper.evaluator.storage._mpi_win_storage import MPIWinStorage

    if not MPI.Is_initialized():
        MPI.Init_thread()

    comm = MPI.COMM_WORLD

    storage_0 = MPIWinStorage(comm)

    # test 0
    with Evaluator.create(
        run_sync,
        method="mpicomm",
        method_kwargs={
            "storage": storage_0,
            "root": 0,
        },
    ) as evaluator:
        if evaluator.is_master:
            evaluator._job_class = HPOJob
            evaluator.submit(
                [
                    {"x": 0},
                    {"x": 1},
                    {"x": 2},
                ]
            )
            job_done = evaluator.gather("ALL")[0]
            assert job_done.metadata["storage_id"] != id(storage_0)
            evaluator.dump_jobs_done_to_csv()

    comm.Barrier()

    # test 1
    # TODO: if I recreate a storage, deadlock
    # something is wrong when (storage_0 gets deleted and creates a deadlock)
    # for example if I use the same name of variable "storage"
    # instead of having storage_0 and storage_1
    storage_1 = MPIWinStorage(comm)

    # serial evaluator
    with Evaluator.create(
        run_sync,
        method="mpicomm",
        method_kwargs={
            "storage": storage_1,
            "root": 0,
        },
    ) as evaluator:
        if evaluator.is_master:
            evaluator._job_class = HPOJob
            evaluator.submit([{"x": i} for i in range(20)])
            job_done = evaluator.gather("BATCH", size=3)[0]
            assert job_done.metadata["storage_id"] != id(storage_1)
            evaluator.dump_jobs_done_to_csv()


@pytest.mark.mpi
def test_mpi_win_storage_with_evaluator():
    command = f"mpirun -np 4 {PYTHON} {SCRIPT} _test_mpi_win_storage_with_evaluator"
    result = dht.run(command, live_output=False)
    assert result.returncode == 0

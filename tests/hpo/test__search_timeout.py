import asyncio
import logging
import multiprocessing
import os
import threading
import time

import pandas as pd

from deephyper.evaluator import Evaluator, JobStatus
from deephyper.hpo import HpProblem, RandomSearch

CPUS = min(4, multiprocessing.cpu_count())


async def run_test_timeout_simple_async(job):
    """An example of async function to be used with the 'serial' evaluator."""
    i = 0
    while True:
        i += 1
        await asyncio.sleep(0.1)
        # The following log line should display "MainThread"
        logging.warning(
            f"working in {threading.current_thread()} of PID={os.getpid()} with status {job.status}"
        )
        if job.status is JobStatus.CANCELLING:
            break
    return i


def run_test_timeout_simple_sync(job):
    """An example of not async function to be used with other evaluators than 'serial'."""
    i = 0
    while True:
        i += 1
        time.sleep(0.1)
        # The following log line should display "ThreadPool-i"
        logging.warning(
            f"working in {threading.current_thread()} of PID={os.getpid()} with status {job.status}"
        )
        if job.status is JobStatus.CANCELLING:
            break
    return i


def test_timeout_simple(tmp_path):
    def run_local_test(run_function, evaluator_method=None, num_workers=1, timeout=1):
        problem = HpProblem()
        problem.add_hyperparameter((0.0, 10.0), "x")

        if type(evaluator_method) is str:
            evaluator = Evaluator.create(
                run_function,
                method=evaluator_method,
                method_kwargs={"num_workers": num_workers},
            )
        else:
            evaluator = run_function

        # Test Timeout without max_evals
        search = RandomSearch(
            problem,
            evaluator,
            random_state=42,
            log_dir=tmp_path,
        )

        t1 = time.time()
        result = search.search(timeout=timeout)
        print(f"{result=}")
        duration = time.time() - t1
        assert duration < timeout + 1
        assert isinstance(result, pd.DataFrame)
        assert result["objective"].iloc[0] >= timeout
        assert len(result) == num_workers

    run_local_test(run_test_timeout_simple_sync, num_workers=1)
    run_local_test(run_test_timeout_simple_async, num_workers=1)

    run_local_test(run_test_timeout_simple_async, "serial", num_workers=1)
    run_local_test(run_test_timeout_simple_async, "serial", num_workers=4)

    run_local_test(run_test_timeout_simple_sync, "thread", num_workers=1)
    run_local_test(run_test_timeout_simple_sync, "thread", num_workers=4)

    run_local_test(run_test_timeout_simple_sync, "process", num_workers=1, timeout=3)
    run_local_test(run_test_timeout_simple_sync, "process", num_workers=CPUS, timeout=3)


def test_timeout_stop_then_continue(tmp_path):
    def run_local_test(run_function, evaluator_method=None, num_workers=1, timeout=1):
        problem = HpProblem()
        problem.add_hyperparameter((0.0, 10.0), "x")

        if type(evaluator_method) is str:
            evaluator = Evaluator.create(
                run_function,
                method=evaluator_method,
                method_kwargs={"num_workers": num_workers},
            )
        else:
            evaluator = run_function

        # Test Timeout without max_evals
        search = RandomSearch(
            problem,
            evaluator,
            random_state=42,
            log_dir=tmp_path,
        )

        # First call
        t1 = time.time()
        result = search.search(timeout=timeout)
        print(f"{result=}")
        duration = time.time() - t1
        assert duration < timeout + 1
        assert isinstance(result, pd.DataFrame)
        assert result["objective"].iloc[0] >= timeout
        assert len(result) == num_workers

        # Second call
        t1 = time.time()
        result = search.search(timeout=timeout)
        print(f"{result=}")
        duration = time.time() - t1

        assert duration < timeout + 1
        assert isinstance(result, pd.DataFrame)
        assert result["objective"].iloc[0] >= timeout
        assert len(result) == num_workers * 2

    run_local_test(run_test_timeout_simple_sync, num_workers=1)
    run_local_test(run_test_timeout_simple_async, num_workers=1)

    run_local_test(run_test_timeout_simple_async, "serial", num_workers=1)
    run_local_test(run_test_timeout_simple_async, "serial", num_workers=4)

    run_local_test(run_test_timeout_simple_sync, "thread", num_workers=1)
    run_local_test(run_test_timeout_simple_sync, "thread", num_workers=4)

    run_local_test(run_test_timeout_simple_sync, "process", num_workers=1, timeout=3)
    run_local_test(run_test_timeout_simple_sync, "process", num_workers=CPUS, timeout=3)


if __name__ == "__main__":
    import logging

    logging.basicConfig(
        # filename="deephyper.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s - %(message)s",
        force=True,
    )
    tmp_path = "/tmp/deephyper_test"

    test_timeout_simple(tmp_path)
    # test_timeout_stop_then_continue(tmp_path)

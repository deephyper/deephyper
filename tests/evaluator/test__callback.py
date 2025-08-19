def test_search_early_stopping_callback(tmp_path):
    from deephyper.evaluator import Evaluator
    from deephyper.hpo import RandomSearch, HpProblem
    from deephyper.evaluator.callback import SearchEarlyStopping

    # Single objective
    problem = HpProblem()
    problem.add_hyperparameter((0.0, 1.0), "x")

    async def run(job):
        return job.parameters["x"]

    evaluator = Evaluator.create(
        run,
        method="serial",
        method_kwargs=dict(
            callbacks=[SearchEarlyStopping(10)],
        ),
    )

    search = RandomSearch(problem, random_state=42, log_dir=tmp_path)
    results = search.search(evaluator, max_evals=1000)

    assert len(results) < 1000

    # Multi-Objective
    problem = HpProblem()
    problem.add_hyperparameter((0.0, 1.0), "x0")
    problem.add_hyperparameter((0.0, 1.0), "x1")

    async def run(job):
        return job.parameters["x0"], job.parameters["x1"]

    evaluator = Evaluator.create(
        run,
        method="serial",
        method_kwargs=dict(
            callbacks=[SearchEarlyStopping(10)],
        ),
    )

    search = RandomSearch(problem, random_state=42, log_dir=tmp_path)
    results = search.search(evaluator, max_evals=1000)

    assert len(results) < 100


def test_tqdm_callback(tmp_path):
    from deephyper.evaluator import Evaluator
    from deephyper.hpo import RandomSearch, HpProblem
    from deephyper.evaluator.callback import TqdmCallback

    # Single objective
    problem = HpProblem()
    problem.add_hyperparameter((0.0, 1.0), "x")

    async def run(job):
        return job.parameters["x"]

    evaluator = Evaluator.create(
        run,
        method="serial",
        method_kwargs=dict(
            callbacks=[TqdmCallback()],
        ),
    )

    search = RandomSearch(problem, random_state=42, log_dir=tmp_path)
    search.search(evaluator, max_evals=100)

    # Multi-Objective
    problem = HpProblem()
    problem.add_hyperparameter((0.0, 1.0), "x0")
    problem.add_hyperparameter((0.0, 1.0), "x1")

    async def run(job):
        return job.parameters["x0"], job.parameters["x1"]

    evaluator = Evaluator.create(
        run,
        method="serial",
        method_kwargs=dict(
            callbacks=[TqdmCallback()],
        ),
    )

    search = RandomSearch(problem, random_state=42, log_dir=tmp_path)
    search.search(evaluator, max_evals=100)


def test_logger_callback(tmp_path):
    from deephyper.evaluator import Evaluator
    from deephyper.hpo import RandomSearch, HpProblem
    from deephyper.evaluator.callback import LoggerCallback

    # Single objective
    problem = HpProblem()
    problem.add_hyperparameter((0.0, 1.0), "x")

    async def run(job):
        return job.parameters["x"]

    evaluator = Evaluator.create(
        run,
        method="serial",
        method_kwargs=dict(
            callbacks=[LoggerCallback()],
        ),
    )

    search = RandomSearch(problem, random_state=42, log_dir=tmp_path)
    search.search(evaluator, max_evals=100)

    # Multi-Objective
    problem = HpProblem()
    problem.add_hyperparameter((0.0, 1.0), "x0")
    problem.add_hyperparameter((0.0, 1.0), "x1")

    async def run(job):
        return job.parameters["x0"], job.parameters["x1"]

    evaluator = Evaluator.create(
        run,
        method="serial",
        method_kwargs=dict(
            callbacks=[LoggerCallback()],
        ),
    )

    search = RandomSearch(problem, random_state=42, log_dir=tmp_path)
    search.search(evaluator, max_evals=100)


def test_csv_logger_callback(tmp_path):
    import os
    import pandas as pd
    import pytest

    from deephyper.evaluator import Job, SerialEvaluator
    from deephyper.evaluator.callback import CSVLoggerCallback

    csv_path = os.path.join(tmp_path, "results.csv")

    # single objective
    configs = [{"x": i} for i in range(10)]

    async def run(job):
        return 7

    evaluator = SerialEvaluator(run, callbacks=[CSVLoggerCallback(csv_path)])
    evaluator._job_class = Job
    evaluator.submit(configs)
    evaluator.gather(type="ALL")
    evaluator.close()
    results = pd.read_csv(csv_path)
    assert list(sorted(results.columns)) == list(
        sorted(
            [
                "p:x",
                "job_id",
                "job_status",
                "m:timestamp_submit",
                "m:timestamp_gather",
                "o:",
            ]
        )
    )
    assert len(results) == 10
    assert results["o:"][3] == 7

    # multi objective
    configs = [{"x": i, "y": i + 10} for i in range(25)]

    async def run(job):
        return 17

    evaluator_multi = SerialEvaluator(run, callbacks=[CSVLoggerCallback(csv_path)])
    evaluator_multi._job_class = Job
    evaluator_multi.submit(configs)
    evaluator_multi.gather(type="ALL")
    evaluator_multi.close()
    results_multi = pd.read_csv(csv_path)
    assert list(sorted(results_multi.columns)) == list(
        sorted(
            [
                "p:x",
                "p:y",
                "job_id",
                "job_status",
                "m:timestamp_submit",
                "m:timestamp_gather",
                "o:",
            ]
        )
    )
    assert len(results_multi) == 25
    assert results_multi["o:"][5] == 17
    assert results_multi["p:x"][10] + 10 == results_multi["p:y"][10]

    # test empty
    os.remove(csv_path)

    configs = [{"x": i} for i in range(0)]

    async def run(job):
        return -1

    results_empty = SerialEvaluator(run, callbacks=[CSVLoggerCallback(csv_path)])
    results_empty._job_class = Job
    results_empty.submit(configs)
    results_empty.gather(type="ALL")
    results_empty.close()

    with pytest.raises(FileNotFoundError):
        results_empty = pd.read_csv(csv_path)


if __name__ == "__main__":
    # test_search_early_stopping_callback(tmp_path=".")
    # test_tqdm_callback(tmp_path=".")
    # test_logger_callback(tmp_path=".")
    test_csv_logger_callback(tmp_path=".")

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

    search = RandomSearch(problem, evaluator, random_state=42, log_dir=tmp_path)
    results = search.search(max_evals=1000)

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

    search = RandomSearch(problem, evaluator, random_state=42, log_dir=tmp_path)
    results = search.search(max_evals=1000)

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

    search = RandomSearch(problem, evaluator, random_state=42, log_dir=tmp_path)
    search.search(max_evals=100)

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

    search = RandomSearch(problem, evaluator, random_state=42, log_dir=tmp_path)
    search.search(max_evals=100)


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

    search = RandomSearch(problem, evaluator, random_state=42, log_dir=tmp_path)
    search.search(max_evals=100)

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

    search = RandomSearch(problem, evaluator, random_state=42, log_dir=tmp_path)
    search.search(max_evals=100)


if __name__ == "__main__":
    # test_search_early_stopping_callback(tmp_path=".")
    # test_tqdm_callback(tmp_path=".")
    test_logger_callback(tmp_path=".")

import logging
from deephyper.hpo import RandomSearch, HpProblem, CBO
from deephyper.evaluator import Evaluator


def test_search_params(caplog):
    """Test search parameters output to logger."""
    caplog.set_level(logging.INFO)

    def run(job):
        return job.parameters["x"]

    problem = HpProblem()
    problem.add_hyperparameter((0.0, 1.0), "x")

    search = RandomSearch(problem, run, checkpoint_history_to_csv=False)
    _ = search.search(max_evals=5)

    assert '"type": "RandomSearch"' in caplog.text
    assert '"max_evals": 5' in caplog.text
    assert '"name": "x"' in caplog.text
    assert '"upper": 1.0' in caplog.text


def run_func(job):
    x = job.parameters["x"]
    b = job.parameters["b"]
    function = job.parameters["function"]
    if function == "linear":
        y = x + b
    elif function == "cubic":
        y = x**3 + b
    return y


def test_search_params_cbo(caplog):
    """Test search parameters output to logger for CBO."""
    caplog.set_level(logging.INFO)

    problem = HpProblem()
    problem.add_hyperparameter((-10.0, 10.0), "x")
    problem.add_hyperparameter((0, 10), "b")
    problem.add_hyperparameter(["linear", "cubic"], "function")

    evaluator = Evaluator.create(
        run_func,
        method="process",
        method_kwargs={
            "num_workers": 2,
        },
    )

    search = CBO(problem, evaluator, random_state=42, checkpoint_history_to_csv=False)
    _ = search.search(max_evals=4)

    assert '"type": "CBO"' in caplog.text
    assert '"max_evals": 4' in caplog.text
    assert '"name": "b"' in caplog.text
    assert '"upper": 10' in caplog.text
    assert '"default_value": "linear"' in caplog.text

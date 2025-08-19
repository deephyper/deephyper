import logging
from deephyper.hpo import RandomSearch, HpProblem, CBO
from deephyper.evaluator import Evaluator


def run(job):
    return job.parameters["x"]


def run_func(job):
    x = job.parameters["x"]
    b = job.parameters["b"]
    function = job.parameters["function"]
    if function == "linear":
        y = x + b
    elif function == "cubic":
        y = x**3 + b
    return y


def test_random_search_params(caplog):
    """Test search parameters output to logger."""
    caplog.set_level(logging.INFO)

    problem = HpProblem()
    problem.add_hyperparameter((0.0, 1.0), "x")

    search = RandomSearch(problem, checkpoint_history_to_csv=False)
    _ = search.search(run, max_evals=5)

    assert "Starting search with RandomSearch" in caplog.text
    assert "Search's problem: " in caplog.text
    assert "Search's evaluator: " in caplog.text
    assert "Running the search for max_evals=5" in caplog.text


def test_cbo_search_params(caplog):
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

    search = CBO(problem, random_state=42, checkpoint_history_to_csv=False)
    _ = search.search(evaluator, max_evals=4)

    assert "Starting search with CBO" in caplog.text
    assert "Search's problem: " in caplog.text
    assert "Search's evaluator: " in caplog.text
    assert "Running the search for max_evals=4" in caplog.text


def test_save_params(tmp_path):
    """Test saving the search parameters to a JSON file."""
    problem = HpProblem()
    problem.add_hyperparameter((0.0, 1.0), "x")

    search = RandomSearch(problem, log_dir=tmp_path, checkpoint_history_to_csv=False)
    _ = search.search(run, max_evals=5)

    search.save_params()
    jsonfile = tmp_path / "params.json"

    assert jsonfile.exists()


def test_get_params():
    """Test getting the search parameters as a dictionary."""
    problem = HpProblem()
    problem.add_hyperparameter((0.0, 1.0), "x")

    search = RandomSearch(problem, checkpoint_history_to_csv=False)
    _ = search.search(run, max_evals=5)

    d = search.get_params()

    assert d["calls"][0]["max_evals"] == 5
    assert d["search"]["problem"]["hyperparameters"][0]["upper"] == 1.0

import os
from deephyper.hpo import RandomSearch, HpProblem


def run(job):
    return job.parameters["x"]


def test_search_without_csv_dump():
    """Execute a search without dumping a CSV file."""
    if os.path.exists("results.csv"):
        os.remove("results.csv")

    problem = HpProblem()
    problem.add_hyperparameter((0.0, 1.0), "x")

    search = RandomSearch(problem, run, checkpoint_history_to_csv=False)

    max_evals = 100
    results = search.search(max_evals)

    assert len(results) == max_evals
    assert "p:x" in results.columns
    assert "objective" in results.columns
    assert not os.path.exists("results.csv")

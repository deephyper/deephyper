import os
from deephyper.hpo import RandomSearch, HpProblem


def test_search_without_csv_dump():
    """Execute a search without dumping a CSV file."""
    if os.path.exists("results.csv"):
        os.remove("results.csv")

    def run(job):
        return job.parameters["x"]

    problem = HpProblem()
    problem.add_hyperparameter((0.0, 1.0), "x")

    search = RandomSearch(problem, run, checkpoint_history_to_csv=False)

    max_evals = 100
    results = search.search(max_evals)

    assert len(results) == max_evals
    assert "p:x" in results.columns
    assert "objective" in results.columns
    assert not os.path.exists("results.csv")


def test_search_pareto():
    """Test pareto efficiency."""
    if os.path.exists("results.csv"):
        os.remove("results.csv")

    def run_multi(job):
        x = job.parameters["x"]
        f1 = (x - 2) ** 2
        f2 = -f1
        return f1, f2

    problem = HpProblem()
    problem.add_hyperparameter((-10.0, 10.0), "x")

    search = RandomSearch(problem, run_multi, checkpoint_history_to_csv=False)
    results = search.search(max_evals=10)

    assert "pareto_efficient" in results.columns
    assert results["pareto_efficient"][0]
    assert not os.path.exists("results.csv")

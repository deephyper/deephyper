def run(job):
    # The suggested parameters are accessible in job.parameters (dict)
    x = job.parameters["x"]
    b = job.parameters["b"]

    if job.parameters["function"] == "linear":
        y = x + b
    elif job.parameters["function"] == "cubic":
        y = x**3 + b

    # Maximization!
    return y


def test_quickstart(tmp_path):
    from deephyper.hpo import HpProblem, CBO
    from deephyper.evaluator import Evaluator

    # define the variable you want to optimize
    problem = HpProblem()
    problem.add_hyperparameter((-10.0, 10.0), "x")  # real parameter
    problem.add_hyperparameter((0, 10), "b")  # discrete parameter
    problem.add_hyperparameter(["linear", "cubic"], "function")  # categorical parameter

    # define the evaluator to distribute the computation
    evaluator = Evaluator.create(
        run,
        method="process",
        method_kwargs={
            "num_workers": 2,
        },
    )

    # define your search and execute it
    search = CBO(problem, evaluator, log_dir=tmp_path, random_state=42)

    results = search.search(max_evals=100)
    print(results)

    assert abs(results.objective.max()) > 1000
    assert "p:x" in results.columns
    assert "p:b" in results.columns
    assert "p:function" in results.columns
    assert len(results) >= 100


if __name__ == "__main__":
    test_quickstart(".")

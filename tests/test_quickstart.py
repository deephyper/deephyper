import pytest


def run(config: dict):
    return -config["x"] ** 2


@pytest.mark.hps
def test_quickstart(tmp_path):
    from deephyper.problem import HpProblem
    from deephyper.search.hps import CBO
    from deephyper.evaluator import Evaluator

    # define the variable you want to optimize
    problem = HpProblem()
    problem.add_hyperparameter((-10.0, 10.0), "x")

    # define the evaluator to distribute the computation
    evaluator = Evaluator.create(
        run,
        method="process",
        method_kwargs={
            "num_workers": 2,
        },
    )

    # define you search and execute it
    search = CBO(problem, evaluator, log_dir=tmp_path, random_state=42)

    results = search.search(max_evals=15)

    assert abs(results.objective.max()) < 0.5
    assert "p:x" in results.columns
    assert len(results) >= 15
    assert len(results) <= 16


if __name__ == "__main__":
    test_quickstart(".")

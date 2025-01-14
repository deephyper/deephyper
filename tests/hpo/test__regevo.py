from ConfigSpace import NotEqualsCondition
from deephyper.hpo import HpProblem, RegularizedEvolution
from deephyper.evaluator import Evaluator


def run(job):
    y = (
        job.parameters["x_int"]
        + job.parameters["x_float"]
        + ord(job.parameters["x_cat"])
        + job.parameters["x_ord"]
        + job.parameters["x_const"]
    )
    return y


def create_problem():
    problem = HpProblem()
    x_int = problem.add_hyperparameter((0, 10), "x_int")
    x_float = problem.add_hyperparameter((0.0, 10.0), "x_float")
    x_cat = problem.add_hyperparameter(["a", "b", "c"], "x_cat")
    x_ord = problem.add_hyperparameter([1, 2, 3], "x_ord")
    x_const = problem.add_hyperparameter(0, "x_const")

    problem.add_condition(NotEqualsCondition(x_int, x_cat, "c"))
    problem.add_condition(NotEqualsCondition(x_float, x_cat, "c"))
    problem.add_condition(NotEqualsCondition(x_ord, x_cat, "c"))
    problem.add_condition(NotEqualsCondition(x_const, x_cat, "c"))
    return problem


def assert_results(results):
    assert "p:x_int" in results.columns
    assert "p:x_float" in results.columns
    assert "p:x_cat" in results.columns
    assert "p:x_ord" in results.columns


def test_centralized_regevo_search(tmp_path):
    problem = create_problem()

    # Test serial evaluation
    search = RegularizedEvolution(
        problem,
        run,
        random_state=42,
        population_size=25,
        sample_size=5,
        log_dir=tmp_path,
        verbose=0,
    )
    results = search.search(max_evals=100)

    assert_results(results)
    assert len(results) == 100

    # Test parallel centralized evaluation
    evaluator = Evaluator.create(
        run_function=run,
        method="thread",
        method_kwargs={"num_workers": 10},
    )
    search = RegularizedEvolution(
        problem,
        evaluator,
        population_size=25,
        sample_size=5,
        log_dir=tmp_path,
        random_state=42,
    )

    results = search.search(max_evals=100, max_evals_strict=True)

    assert_results(results)
    assert len(results) >= 100


if __name__ == "__main__":
    test_centralized_regevo_search(tmp_path="/tmp/deephyper_test")

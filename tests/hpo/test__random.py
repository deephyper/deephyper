import pytest

from ConfigSpace import NotEqualsCondition
from deephyper.hpo import HpProblem, RandomSearch
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


def assert_results(results, max_evals_strict=False):
    if max_evals_strict:
        assert len(results) == 100
    else:
        assert len(results) >= 100
    assert "p:x_int" in results.columns
    assert "p:x_float" in results.columns
    assert "p:x_cat" in results.columns
    assert "p:x_ord" in results.columns


@pytest.mark.fast
def test_centralized_random_search(tmp_path):

    problem = create_problem()

    # Test serial evaluation
    search = RandomSearch(problem, run, random_state=42, log_dir=tmp_path, verbose=0)
    results = search.search(max_evals=100)

    assert_results(results)

    # Test parallel centralized evaluation
    evaluator = Evaluator.create(
        run_function=run,
        method="thread",
        method_kwargs={"num_workers": 10},
    )
    search = RandomSearch(problem, evaluator, random_state=42, log_dir=tmp_path)

    results = search.search(max_evals=100)

    assert_results(
        results,
    )


@pytest.mark.fast
@pytest.mark.redis
def test_centralized_random_search_redis_storage(tmp_path):
    from deephyper.evaluator.storage import RedisStorage

    storage = RedisStorage()

    problem = create_problem()

    evaluator = Evaluator.create(
        run_function=run,
        method="process",
        method_kwargs={"storage": storage, "num_workers": 10},
    )

    search = RandomSearch(problem, evaluator, random_state=42, log_dir=tmp_path)

    results = search.search(max_evals=100)

    assert_results(results)


def launch_serial_search_with_redis_storage(search_id, search_seed, is_master=False):
    from deephyper.evaluator.storage import RedisStorage

    storage = RedisStorage().connect()

    problem = create_problem()

    evaluator = Evaluator.create(
        run_function=run,
        method="serial",
        method_kwargs={"storage": storage, "num_workers": 1, "search_id": search_id},
    )

    log_dir = "." if is_master else f"/tmp/deephyper_search_{search_seed}"
    search = RandomSearch(problem, evaluator, random_state=search_seed, log_dir=log_dir)

    def dump_evals(*args, **kwargs):
        pass

    max_evals = 100
    results = None
    max_evals_strict = True
    if is_master:
        results = search.search(max_evals=max_evals, max_evals_strict=max_evals_strict)
    else:
        evaluator.dump_jobs_done_to_csv = dump_evals
        search.search(max_evals=max_evals, max_evals_strict=max_evals_strict)

    return results


@pytest.mark.fast
@pytest.mark.redis
def test_decentralized_random_search_redis_storage():
    from multiprocessing import Pool
    from deephyper.evaluator.storage import RedisStorage

    storage = RedisStorage().connect()
    search_id = storage.create_new_search()

    # Master
    with Pool(processes=4) as pool:
        results = pool.starmap(
            launch_serial_search_with_redis_storage,
            [(search_id, i, i == 0) for i in range(4)],
        )
    assert_results(results[0], max_evals_strict=False)


if __name__ == "__main__":
    test_centralized_random_search()
    test_centralized_random_search_redis_storage()
    test_decentralized_random_search_redis_storage()

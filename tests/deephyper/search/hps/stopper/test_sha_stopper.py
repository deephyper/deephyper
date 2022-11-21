from deephyper.evaluator import Evaluator, RunningJob, profile
from deephyper.evaluator.callback import TqdmCallback
from deephyper.evaluator.storage import RedisStorage
from deephyper.problem import HpProblem
from deephyper.search.hps import CBO
from deephyper.stopper import AsyncSuccessiveHalvingStopper


@profile
def run(job: RunningJob) -> dict:

    max_budget = 50
    objective_i = 0

    for budget_i in range(1, max_budget + 1):
        objective_i += job["x"]

        job.record(budget_i, objective_i)
        if job.stopped():
            break

    # obs = {b: o for b, o in zip(*job.observations)}
    # print(f"{obs=}")
    return {
        "objective": job.observations,
        "metadata": {"budget": budget_i, "stopped": budget_i < max_budget},
    }


if __name__ == "__main__":
    # define the variable you want to optimize
    problem = HpProblem()
    problem.add_hyperparameter((0.0, 10.0), "x")

    storage = RedisStorage()
    storage.connect()
    storage._redis.flushdb()
    evaluator = Evaluator.create(
        run,
        method="process",
        method_kwargs={"storage": storage, "callbacks": []},
    )
    stopper = AsyncSuccessiveHalvingStopper(min_budget=1, reduction_factor=3)
    search = CBO(problem, evaluator, surrogate_model="RF", stopper=stopper)

    results = search.search(max_evals=20)
    print(results)

    # from pprint import pprint

    # storage = search._evaluator._storage
    # pprint(storage.load_search(search._evaluator._search_id))

    # rung_0 = storage.load_metadata_from_all_other_jobs("0.9", "completed_rung_0")
    # pprint(f"{rung_0=}")

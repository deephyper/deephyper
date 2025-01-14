from deephyper.evaluator import Evaluator
from deephyper.evaluator.callback import TqdmCallback
from deephyper.hpo import CBO


def execute_centralized_bo_with_share_memory(
    problem,
    run_function,
    run_function_kwargs,
    storage,
    search_id,
    search_random_state,
    log_dir,
    num_workers,
    is_master,
    kappa,
    search_kwargs,
    timeout,
):
    evaluator = Evaluator.create(
        run_function,
        method="thread",
        method_kwargs={
            "num_workers": num_workers,
            "storage": storage,
            "search_id": search_id,
            "callbacks": [TqdmCallback()] if is_master else [],
            "run_function_kwargs": run_function_kwargs,
        },
    )

    search_kwargs["kappa"] = kappa
    search_kwargs["random_state"] = search_random_state
    search = CBO(problem, evaluator, log_dir=log_dir, **search_kwargs)

    def dummy(*args, **kwargs):
        pass

    results = None
    if is_master:
        results = search.search(timeout=timeout)
    else:
        # for concurrency reasons this is important to override these functions
        evaluator.dump_jobs_done_to_csv = dummy
        search.extend_results_with_pareto_efficient = dummy

        search.search(timeout=timeout)

    return results

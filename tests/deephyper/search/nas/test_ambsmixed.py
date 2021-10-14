
#! with subprocess be carefull about this IF statement otherwise it will enter in a
#! infinite loop
if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.DEBUG)

    from deephyper.search.nas import AMBSMixed
    from deephyper.evaluator import Evaluator

    from deephyper.benchmark.nas.linearRegHybrid import Problem
    from deephyper.nas.run import run_debug_slow

    evaluator = Evaluator.create(
        run_debug_slow, method="subprocess", method_kwargs={"num_workers": 1}
    )

    search = AMBSMixed(Problem, evaluator)

    search.search(max_evals=10)

    search.search(max_evals=100, timeout=1)

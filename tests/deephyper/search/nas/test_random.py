
#! with subprocess be carefull about this IF statement otherwise it will enter in a
#! infinite loop
if __name__ == "__main__":
    import os
    import logging

    logging.basicConfig(level=logging.DEBUG)

    from deephyper.search.nas.random import Random
    from deephyper.evaluator.evaluate import Evaluator

    from deephyper.benchmark.nas.linearReg import Problem
    from deephyper.nas.run.quick import run
    print("HERE")
    evaluator = Evaluator.create(
        run, method="ray", method_kwargs={"num_cpus": 1}
    )

    search = Random(Problem, evaluator)

    search.search(max_evals=10)

    # search.search(max_evals=100, timeout=1)

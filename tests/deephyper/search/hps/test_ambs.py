def run(hp):
    return hp["x"]


#! with subprocess be carefull about this IF statement otherwise it will enter in a
#! infinite loop
if __name__ == "__main__":
    import os
    import logging

    logging.basicConfig(level=logging.DEBUG)

    from deephyper.problem import HpProblem
    from deephyper.search.hps import AMBS
    from deephyper.evaluator.evaluate import Evaluator
    from deephyper.evaluator.callback import ProfilingCallback

    import matplotlib.pyplot as plt


    problem = HpProblem()
    problem.add_hyperparameter((0.0, 10.0), "x")

    evaluator = Evaluator.create(
        run, method="ray", method_kwargs={
            # "num_cpus": 4, 
            "num_workers": 4,
            "callbacks":[ProfilingCallback()]
        }
    )

    search = AMBS(problem, evaluator)

    if os.path.exists("results.csv"):
        search.fit_surrogate("results.csv")

    search.search(max_evals=100)

    profile = evaluator._callbacks[0].profile
    print(profile)

    plt.figure()
    plt.step(profile.timestamp, profile.n_jobs_running)
    plt.ylim(top=5)
    plt.show()

    # search.search(max_evals=100, timeout=1)

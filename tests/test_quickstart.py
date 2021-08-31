def run(config: dict):
    return -config["x"]**2


# Necessary IF statement otherwise it will enter in a infinite loop
# when loading the 'run' function from a subprocess
if __name__ == "__main__":
    from deephyper.problem import HpProblem
    from deephyper.search.hps import AMBS
    from deephyper.evaluator.evaluate import Evaluator

    # define the variable you want to optimize
    problem = HpProblem()
    problem.add_hyperparameter((-10.0, 10.0), "x")

    # define the evaluator to distribute the computation
    evaluator = Evaluator.create(
        run,
        method="subprocess",
        method_kwargs={
            "num_workers": 2,
        },
    )

    # define you search and execute it
    search = AMBS(problem, evaluator)

    results = search.search(max_evals=100)
    print(results)
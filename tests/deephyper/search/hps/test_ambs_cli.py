"""
* deephyper hps ambs --problem test_cli.problem --evaluator thread --run-function test_cli.run --max-evals 100 --timeout 10

* deephyper hps ambs --problem test_ambs_cli.problem --evaluator thread --run-function test_ambs_cli.run --max-evals 100 --timeout 10 --kappa 1.96 --n-jobs 2 --verbose 1

"""
# from deephyper.problem import HpProblem

# def run(hp):
#     return hp["x"]

# problem = HpProblem()
# problem.add_hyperparameter((0.0, 10.0), "x")

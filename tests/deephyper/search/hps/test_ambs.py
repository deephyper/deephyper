import os
import logging

logging.basicConfig(level=logging.DEBUG)

from deephyper.problem import HpProblem
from deephyper.search.hps import AMBS
from deephyper.evaluator.evaluate import Evaluator

problem = HpProblem()
problem.add_hyperparameter((0.0, 10.0), "x")

def run(hp):
    return hp["x"]

evaluator = Evaluator.create(run, method="ray", method_kwargs={"num_cpus": 8})

search = AMBS(problem, evaluator)

if os.path.exists("results.csv"):
    search.fit_surrogate("results.csv")

search.search(max_evals=10)

# search.search(max_evals=100, timeout=1)
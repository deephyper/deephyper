"""
* deephyper nas agebo --problem test_agebo_cli.problem --evaluator thread --run-function test_agebo_cli.run --max-evals 100 --timeout 10

* deephyper nas agebo --problem test_agebo_cli.problem --evaluator thread --run-function test_agebo_cli.run --max-evals 100 --timeout 10 --kappa 1.96 --n-jobs 2 --verbose 1

"""

from deephyper.benchmark.nas.linearRegHybrid import Problem as problem
from deephyper.nas.run.quick import run
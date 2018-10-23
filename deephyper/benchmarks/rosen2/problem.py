import random
from deephyper.benchmarks import HpProblem
NDIM = 2


Problem = HpProblem()
def_values = [random.uniform(-3.0, 4.0) for i in range(NDIM)]
for i in range(1, 1+NDIM):
    dim = f"x{i}"
    Problem.add_dim(dim, (-3.0, 4.0), p_default=def_values[i])

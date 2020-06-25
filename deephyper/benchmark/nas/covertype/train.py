"""
KMP_BLOCK_TIME=0


horovodrun -np 4 python -m deephyper.benchmark.nas.covertype.train
"""

from deephyper.benchmark.nas.covertype.problem import Problem
from deephyper.search.nas.model.run.horovod import run

config = Problem.space

config["hyperparameters"]["num_epochs"] = 100
config["hyperparameters"]["learning_rate"] = 0.00335351825


config["arch_seq"] = [
    22,
    0,
    12,
    0,
    1,
    14,
    1,
    1,
    0,
    17,
    1,
    1,
    0,
    24,
    1,
    1,
    1,
    15,
    1,
    0,
    1,
    30,
    0,
    1,
    1,
    19,
    1,
    1,
    1,
    3,
    1,
    1,
    1,
    26,
    0,
    1,
    0,
]

run(config)

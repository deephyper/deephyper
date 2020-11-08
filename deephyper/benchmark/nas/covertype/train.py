"""
KMP_BLOCK_TIME=0


horovodrun -np 4 python -m deephyper.benchmark.nas.covertype.train
"""

from deephyper.benchmark.nas.covertype.problem import Problem
from deephyper.nas.run.horovod import run

config = Problem.space

config["hyperparameters"]["num_epochs"] = 100
config["hyperparameters"]["learning_rate"] = 0.00333302975


config["arch_seq"] = [
    22,
    0,
    22,
    0,
    0,
    27,
    1,
    0,
    0,
    22,
    0,
    0,
    0,
    17,
    1,
    0,
    0,
    27,
    0,
    0,
    0,
    19,
    1,
    0,
    0,
    30,
    0,
    1,
    0,
    9,
    1,
    0,
    1,
    17,
    1,
    1,
    0,
]

run(config)

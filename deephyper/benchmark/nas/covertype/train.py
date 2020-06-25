"""
KMP_BLOCK_TIME=0


horovodrun -np 4 python -m deephyper.benchmark.nas.covertype.train
"""

from deephyper.benchmark.nas.covertype.problem import Problem
from deephyper.search.nas.model.run.horovod import run

config = Problem.space

config["hyperparameters"]["num_epochs"] = 100
config["hyperparameters"]["learning_rate"] = 0.01


config["arch_seq"] = [
    19,
    0,
    24,
    0,
    0,
    30,
    1,
    0,
    1,
    9,
    1,
    0,
    0,
    27,
    1,
    0,
    1,
    28,
    1,
    0,
    1,
    20,
    0,
    0,
    0,
    11,
    0,
    0,
    1,
    27,
    1,
    1,
    0,
    0,
    0,
    0,
    0,
]

run(config)

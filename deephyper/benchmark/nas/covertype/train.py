"""
KMP_BLOCK_TIME=0


horovodrun -np 4 python -m deephyper.benchmark.nas.covertype.train
"""

from deephyper.benchmark.nas.covertype.problem import Problem
from deephyper.search.nas.model.run.horovod import run

config = Problem.space

config["arch_seq"] = [
    16,
    0,
    11,
    0,
    0,
    10,
    1,
    1,
    1,
    29,
    1,
    1,
    1,
    24,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    19,
    1,
    0,
    0,
    26,
    1,
    1,
    0,
    29,
    1,
    1,
    0,
    23,
    0,
    0,
    0,
]

run(config)

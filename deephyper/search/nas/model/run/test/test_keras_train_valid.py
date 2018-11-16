from random import random

from deephyper.search.nas.run.keras_train_valid import run
from deephyper.benchmark.nas.toy.pb_keras_class import Problem as pb_class

config = pb_class.space
# config['arch_seq'] = [random() for i in range(100)]
config['arch_seq'] = [0.375,
            0.25,
            0.25,
            0.0,
            0.25,
            0.0,
            0.625,
            0.0,
            0.375,
            0.375,
            0.0,
            0.375,
            0.0,
            0.375,
            0.0]
run(config)

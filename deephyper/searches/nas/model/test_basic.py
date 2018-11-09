import os
import signal
import sys
import time
from importlib import import_module
from pprint import pformat
from random import random

import numpy as np
import tensorflow as tf

import deephyper.searches.nas.model.arch as a
# from deephyper.benchmarks.candleNT3Nas.problem import Problem
# from deephyper.benchmarks.candleTC1Nas.problem import Problem
# from deephyper.benchmarks.mnist1DNas.problem import Problem
from deephyper.benchmarks.polynome2RegNas.problem import Problem
from deephyper.searches import util
from deephyper.searches.nas.model.trainer import BasicTrainer
# from deephyper.benchmarks.linearRegNas.problem import Problem
# from deephyper.benchmarks.ackleyRegNas.problem import Problem
from deephyper.searches.nas.run.nas_structure_raw import run

t1 = time.time()




logger = util.conf_logger('deephyper.searches.nas.model.test_basic')

def main(config):
    # config['arch_seq'] = [float(e) for e in np.random.choice(2, 200)]
    config['arch_seq'] = [
            0.375,
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
            0.0
        ]
    architecture = config['arch_seq']
    print(f'arch_seq: {architecture}')

    result = run(config)

    print('OUTPUT: ', result)
    return result

if __name__ == '__main__':
    pb = Problem
    param_dict = pb.space
    main(param_dict)

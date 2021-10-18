"""The :func:`deephyper.nas.run.quick2.run` function is a function used to check the good behaviour of a mixed hyperparameter and neural architecture search algorithm. It will simply return an objective combining the sum of the scalar values encoding a neural architecture in the ``config["arch_seq"]`` key then divide this sum by the ``batch_size`` hyperparameter and scale it by the ``learning_rate`` hyperparameter::

    (sum(arch_seq) + randn() * noise_level) / batch_size * learning_rate
"""

import numpy as np


def run_debug_hp_arch(config: dict) -> float:
    noise_level = 1.0
    lr = config["hyperparameters"]["learning_rate"]
    bs = config["hyperparameters"]["batch_size"]
    return (sum(config["arch_seq"]) + np.random.randn() * noise_level) / bs * lr

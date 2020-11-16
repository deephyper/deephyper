import traceback

import numpy as np
import tensorflow as tf
from tensorflow import keras

from deephyper.search import util
from .alpha import run as run_alpha

logger = util.conf_logger("deephyper.search.nas.run")


def run(config):
    seed = config["seed"]
    repeat = 2
    if seed is not None:
        np.random.seed(seed)
        seeds = np.random.randint(0, 2 ** 32 - 1, repeat)

    res_list = []
    for i in range(repeat):
        tf.keras.backend.clear_session()
        if seed is not None:
            config["seed"] = seeds[i]
        res = run_alpha(config)
        res_list.append(res)

    return max(res_list)

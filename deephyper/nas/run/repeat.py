"""The :func:`deephyper.nas.run.repeat.run` function is a function used to repeat ``N`` times the evaluation performed with the :func:`deephyper.nas.run.alpha.run`. To use this evaluation function the ``repeat=N`` argument has to be defined in the list of hyperparameters of the :func:`deephyper.problem.NaProblem` object:

.. code-block:: python

    Problem.hyperparameters(
        ...,
        repeat=N
        ...
    )
"""
import numpy as np
import tensorflow as tf
import logging

from deephyper.search import util
from deephyper.nas.run.alpha import run as run_alpha

logger = logging.getLogger(__name__)


def run(config: dict) -> float:
    seed = config["seed"]
    repeat = config["hyperparameters"].get("repeat", 1)
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

"""The :func:`deephyper.nas.run.quick_random.run` function is a function used to check the good behaviour of an hyperparameter or neural architecture search algorithm. It will simply return an objective of the sum of hyperparameters combined with a random sample to check the good reproducibility of DeepHyper experiments while setting a random seed in the problem definition.
"""
import numpy as np


def run_debug(config: dict) -> float:
    random = np.random.RandomState(config.get("seed"))
    if "arch_seq" in config:
        return sum(config["arch_seq"]) + random.random()
    else:
        return sum(config.values()) + random.random()

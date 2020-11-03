import numpy as np


def run(config):
    noise_level = 1.0
    lr = config["hyperparameters"]["learning_rate"]
    bs = config["hyperparameters"]["batch_size"]
    return (sum(config["arch_seq"]) + np.random.randn() * noise_level) / bs * lr

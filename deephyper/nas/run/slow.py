"""The :func:`deephyper.nas.run.quick.run` function is a function used to check the good behaviour of a neural architecture search algorithm. It will simply return the sum of the scalar values encoding a neural architecture in the ``config["arch_seq"]`` key.
"""
import time

def run(config: dict) -> float:
    time.sleep(1)
    return sum(config['arch_seq'])

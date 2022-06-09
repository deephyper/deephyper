"""The :func:`deephyper.nas.run.quick.run` function is a function used to check the good behaviour of a neural architecture search algorithm. It will simply return the sum of the scalar values encoding a neural architecture in the ``config["arch_seq"]`` key.
"""


def run_debug_arch(config: dict) -> float:
    return sum(config["arch_seq"])

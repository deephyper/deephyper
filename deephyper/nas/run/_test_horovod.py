"""The :func:`deephyper.nas.run.test_horovod.run` function is used to check the good behaviour of a call made by within an Horovod context.
"""
import os
import time
import random

import horovod.tensorflow as hvd


def run(config: dict) -> float:
    """Using the stateless `run` method, a function can take in any args or kwargs"""

    print("hvd init...", end="", flush=True)
    hvd.init()
    print("OK", flush=True)

    print(
        "hvd rank: ",
        hvd.rank(),
        " - CUDA_VISIBLE: ",
        os.environ.get("CUDA_VISIBLE_DEVICES"),
    )

    duration = random.choice([3, 4, 5])
    print(f"sleep {duration}...", end="", flush=True)
    time.sleep(duration)
    print("OK", flush=True)

    return 0

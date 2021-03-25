import os
import time
import random

import horovod.tensorflow as hvd


def run(config: dict) -> float:
    """Using the stateless `run` method, a function can take in any args or kwargs"""

    hvd.init()

    print(
        "hvd rank: ",
        hvd.rank(),
        " - CUDA_VISIBLE: ",
        os.environ.get("CUDA_VISIBLE_DEVICES"),
    )

    duration = random.choice([3,4,5])
    time.sleep(duration)

    return 0
import os

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

    return 0
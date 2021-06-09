"""
Example command line::

    deephyper hps ambs --evaluator subprocess --problem deephyper.sklearn.regressor.Problem --run deephyper.benchmark.nas.covertype.problem_hps.run
"""
import os
import numpy as np

from deephyper.sklearn.regressor import run as sklearn_run


HERE = os.path.dirname(os.path.abspath(__file__))


def load_data():
    exp = 1
    data = np.load(
        os.path.join(HERE, f"DATA/nas_dataset_exp{exp}.npy"), allow_pickle=True
    )[()]
    X, y = data["X"], data["y"]
    print(np.shape(X))
    print(np.shape(y))
    return X, y


def run(config):
    return sklearn_run(config, load_data)


if __name__ == "__main__":
    # load_data()
    from deephyper.sklearn.regressor import Problem
    config = Problem.space.sample_configuration()
    config = dict(config)
    r2 = run(config)
    print(config)
    print("r2: ", r2)

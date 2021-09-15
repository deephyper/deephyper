import numpy as np

from deephyper.sklearn.classifier.autosklearn1.problem import Problem
from deephyper.sklearn.classifier.autosklearn1.run import run as autosklearn_run


def load_data():
    from sklearn.datasets import load_iris

    X, y = load_iris(return_X_y=True)
    print(np.shape(X))
    print(np.shape(y))
    return X, y


def run(config):
    return autosklearn_run(config, load_data)


if __name__ == "__main__":
    config = Problem.space.sample_configuration()
    config = dict(config)
    acc = run(config)
    print(config)
    print("acc: ", acc)

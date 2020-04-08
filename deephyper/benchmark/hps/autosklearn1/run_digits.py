import numpy as np

from deephyper.search.hps.automl.classifier import autosklearn1


def load_data():
    from sklearn.datasets import load_digits

    X, y = load_digits(return_X_y=True)
    print(np.shape(X))
    print(np.shape(y))
    return X, y


def run(config):
    return autosklearn1.run(config, load_data)


if __name__ == "__main__":
    config = autosklearn1.Problem.space.sample_configuration()
    config = dict(config)
    acc = run(config)
    print(config)
    print("acc: ", acc)

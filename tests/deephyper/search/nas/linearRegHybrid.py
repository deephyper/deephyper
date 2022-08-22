from tabnanny import verbose
import numpy as np

from deephyper.problem import NaProblem
from deephyper.nas.spacelib.tabular import OneLayerSpace


def load_data(dim=10, verbose=0):
    """
    Generate data for linear function -sum(x_i).

    Return:
        Tuple of Numpy arrays: ``(train_X, train_y), (valid_X, valid_y)``.
    """
    rng = np.random.RandomState(42)
    size = 100000
    prop = 0.80
    a, b = 0, 100
    d = b - a
    x = np.array([a + rng.random(dim) * d for i in range(size)])
    y = np.array([[np.sum(v)] for v in x])

    sep_index = int(prop * size)
    train_X = x[:sep_index]
    train_y = y[:sep_index]

    valid_X = x[sep_index:]
    valid_y = y[sep_index:]

    if verbose:
        print(f"train_X shape: {np.shape(train_X)}")
        print(f"train_y shape: {np.shape(train_y)}")
        print(f"valid_X shape: {np.shape(valid_X)}")
        print(f"valid_y shape: {np.shape(valid_y)}")
    return (train_X, train_y), (valid_X, valid_y)


Problem = NaProblem()
Problem.load_data(load_data)
Problem.search_space(OneLayerSpace)
Problem.hyperparameters(
    batch_size=Problem.add_hyperparameter((1, 100), "batch_size"),
    learning_rate=Problem.add_hyperparameter(
        (1e-4, 1e-1, "log-uniform"), "learning_rate"
    ),
    optimizer=Problem.add_hyperparameter(["adam", "nadam", "rmsprop"], "optimizer"),
    num_epochs=1,
)
Problem.loss("mse")
Problem.metrics(["r2"])
Problem.objective("val_r2")

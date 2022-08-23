import numpy as np


def load_data(dim=10, verbose=0):
    """
    Generate data for linear function -sum(x_i).

    Return:
        Tuple of Numpy arrays: ``(train_X, train_y), (valid_X, valid_y)``.
    """
    rng = np.random.RandomState(42)
    size = 10000
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


if __name__ == "__main__":
    load_data(verbose=1)

import os
import numpy as np
from deephyper.benchmark.benchmark_functions_wrappers import linear_

np.random.seed(2018)

def load_data(dim=10):
    """
    Generate data for linear function -sum(x_i).

    Return:
        Tuple of Numpy arrays: ``(train_X, train_y), (valid_X, valid_y)``.
    """
    # size = 100000
    size = 100
    prop = 0.80
    f, (a, b), _ = linear_()
    d = b - a
    x = np.array([a + np.random.random(dim) * d for i in range(size)])
    y = np.array([[f(v)] for v in x])

    sep_index = int(prop * size)

    sep_inputs = dim//2 # we want two different inputs
    tX0, tX1 = x[:sep_index, :sep_inputs], x[:sep_index, sep_inputs:]
    vX0, vX1 = x[sep_index:, :sep_inputs], x[sep_index:, sep_inputs:]

    ty = y[:sep_index]
    vy = y[sep_index:]

    print(f'tX0 shape: {np.shape(tX0)} | tX1 shape: {np.shape(tX1)}')
    print(f'ty shape: {np.shape(ty)}')
    print(f'vX0 shape: {np.shape(vX0)} | vX1 shape: {np.shape(vX1)}')
    print(f'vy shape: {np.shape(vy)}')
    return ([tX0, tX1], ty), ([vX0, vX1], vy)

if __name__ == '__main__':
    load_data()

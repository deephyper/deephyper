import os
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

np.random.seed(2018)

def load_data(dest=None):
    """
    Generate data for cosinus function.
    Returns Tuple of Numpy arrays: `(train_X, train_y), (valid_X, valid_y)`.
    """
    size = 10000
    prop = 0.85
    arr = np.arange(size)
    np.random.shuffle(arr)

    x = np.linspace(0, 10, size)
    y = np.copy(x)

    x = x[arr]
    y = y[arr]

    x = np.reshape(x, (size, 1))
    y = np.reshape(y, (size, 1))

    sep_index = int(prop * size)
    train_X = x[:sep_index]
    train_y = y[:sep_index]

    valid_X = x[sep_index:]
    valid_y = y[sep_index:]

    print(f'train_X shape: {np.shape(train_X)}')
    print(f'train_y shape: {np.shape(train_y)}')
    print(f'valid_X shape: {np.shape(valid_X)}')
    print(f'valid_y shape: {np.shape(valid_y)}')
    return (train_X, train_y), (valid_X, valid_y)

if __name__ == '__main__':
    load_data()

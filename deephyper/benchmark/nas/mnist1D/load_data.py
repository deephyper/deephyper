import os
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

def load_data():
    """Loads the MNIST dataset.
    Returns Tuple of Numpy arrays: `(train_X, train_y), (valid_X, valid_y)`.
    """
    dest = HERE+'/DATA'

    mnist = input_data.read_data_sets(dest, one_hot=True)

    train_X = mnist.train.images / 255
    train_y = mnist.train.labels

    valid_X = mnist.validation.images / 255
    valid_y = mnist.validation.labels

    print(f'train_X shape: {np.shape(train_X)}')
    print(f'train_y shape: {np.shape(train_y)}')
    print(f'valid_X shape: {np.shape(valid_X)}')
    print(f'valid_y shape: {np.shape(valid_y)}')
    return (train_X, train_y), (valid_X, valid_y)

if __name__ == '__main__':
    load_data()

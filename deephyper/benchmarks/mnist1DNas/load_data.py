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

    train_X = mnist.train.images
    train_y = mnist.train.labels

    valid_X = mnist.validation.images
    valid_y = mnist.validation.labels

    return (train_X, train_y), (valid_X, valid_y)

if __name__ == '__main__':
    load_data()

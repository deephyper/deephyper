from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

def load_data(dest):
    """Loads the MNIST dataset.
    Returns Tuple of Numpy arrays: `(train_X, train_Y), (valid_X, valid_Y)`.
    """

    mnist = input_data.read_data_sets(dest, one_hot=True)

    train_X = mnist.train.images,
    train_Y = np.argmax(mnist.train.labels, axis=1),
    valid_X = mnist.validation.images,
    valid_Y = np.argmax(mnist.validation.labels, axis=1)

    return (train_X, train_Y), (valid_X, valid_Y)

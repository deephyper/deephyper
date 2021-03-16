import os

import tensorflow as tf
import numpy as np

from tensorflow.keras.utils import to_categorical

HERE = os.path.dirname(os.path.abspath(__file__))

np.random.seed(2018)


def load_data(prop=0.1):
    """Loads the MNIST dataset.
    Returns Tuple of Numpy arrays: `(train_X, train_y), (valid_X, valid_y)`.
    """
    dest = "mnist.npz"

    (train_X, train_y), (valid_X, valid_y) = tf.keras.datasets.mnist.load_data(path=dest)

    train_X = train_X / 255
    valid_X = valid_X / 255

    train_X = train_X.reshape(len(train_X), -1)
    valid_X = valid_X.reshape(len(valid_X), -1)

    num_classes = len(np.unique(train_y))
    train_y = to_categorical(train_y, num_classes)
    valid_y = to_categorical(valid_y, num_classes)

    # Subset selection for training
    train_size = np.shape(train_X)[0]
    limit = int(train_size * prop)
    indexes = np.array([i for i in range(train_size)])
    np.random.shuffle(indexes)
    train_X = train_X[indexes[:limit]]
    train_y = train_y[indexes[:limit]]

    print(f"train_X shape: {np.shape(train_X)}")
    print(f"train_y shape: {np.shape(train_y)}")
    print(f"valid_X shape: {np.shape(valid_X)}")
    print(f"valid_y shape: {np.shape(valid_y)}")
    return (train_X, train_y), (valid_X, valid_y)


if __name__ == "__main__":
    load_data()

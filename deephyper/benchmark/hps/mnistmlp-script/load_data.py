import os
from keras.datasets import mnist

def load_data():
    """Loads the MNIST dataset.
    Returns Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    directory = os.path.join(here, "DATA")
    if not os.path.exists(directory):
        os.makedirs(directory)

    dest = os.path.join(here, 'DATA', 'mnist.npz')

    (x_train, y_train), (x_test, y_test) = mnist.load_data(dest)
    return (x_train, y_train), (x_test, y_test)

if __name__ == '__main__':
    load_data()
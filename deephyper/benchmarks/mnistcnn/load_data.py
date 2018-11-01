from keras.utils.data_utils import get_file
import numpy as np

def load_data(origin, dest):
    """Loads the MNIST dataset.
    Returns Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    path = get_file('mnist.npz',
                    origin='file://'+origin,
                    cache_subdir=dest,
                    file_hash='8a61469f7ea1b51cbae51d4f78837e45'
                   )
    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)

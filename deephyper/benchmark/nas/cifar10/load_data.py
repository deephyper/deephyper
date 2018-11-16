from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import numpy as np

from keras.datasets.cifar import load_batch
import keras.backend as K
from keras.utils.data_utils import get_file

HERE = os.path.dirname(os.path.abspath(__file__))

def load_data(dest=None):
    """Loads CIFAR10 dataset.
    Returns:
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    origin = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    dest = HERE+'/DATA'
    if dest is None:
        dest = '/projects/datascience/regele/deephyper/benchmark/cifar10Nas/DATA'
    else:
        dest = os.path.abspath(os.path.expanduser(dest))

    print(f"getfile(origin={origin}, dest={dest})")

    path = get_file('cifar-10-batches-py', origin=origin, untar=True,
                    cache_subdir=dest)

    num_train_samples = 50000

    train_X = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
    train_y = np.empty((num_train_samples,), dtype='uint8')

    for i in range(1, 6):
        fpath = os.path.join(path, 'data_batch_' + str(i))
        (train_X[(i - 1) * 10000:i * 10000, :, :, :],
         train_y[(i - 1) * 10000:i * 10000]) = load_batch(fpath)

    fpath = os.path.join(path, 'test_batch')
    test_X, test_y = load_batch(fpath)

    train_y = np.reshape(train_y, (len(train_y)))
    test_y = np.reshape(test_y, (len(test_y)))

    train_X = np.true_divide(train_X, 255)
    test_X = np.true_divide(test_X, 255)

    if K.image_data_format() == 'channels_last':
        train_X = train_X.transpose(0, 2, 3, 1)
        test_X = test_X.transpose(0, 2, 3, 1)
    return (train_X, train_y), (test_X, test_y)

if __name__ == '__main__':
    (train_X, train_y), (test_X, test_Y) = load_data()
    print(train_X[0])
    print(f'train_X shape = {np.shape(train_X)}')
    print(f'train_y shape = {np.shape(train_y)}')
    print(f'test_X shape = {np.shape(test_X)}')
    print(f'test_y shape = {np.shape(test_y)}')
#load_data('cifar10_data')

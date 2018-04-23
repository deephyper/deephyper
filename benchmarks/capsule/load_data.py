from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import numpy as np

from keras.datasets.cifar import  load_batch
import keras.backend as K
from keras.utils.data_utils import get_file

def load_data(origin, dest):
    """Loads CIFAR10 dataset.
    Returns:
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    #origin = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    if dest is None:
        dest = 'datasets'
    else:
        dest = os.path.abspath(os.path.expanduser(dest))
    
    print(f"getfile(origin={origin}, dest={dest})")

    path = get_file('cifar-10-batches-py', origin='file://'+origin, untar=True,
                    cache_subdir=dest)

    num_train_samples = 50000
  
    x_train = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
    y_train = np.empty((num_train_samples,), dtype='uint8')
  
    for i in range(1, 6):
        fpath = os.path.join(path, 'data_batch_' + str(i))
        (x_train[(i - 1) * 10000:i * 10000, :, :, :],
         y_train[(i - 1) * 10000:i * 10000]) = load_batch(fpath)
  
    fpath = os.path.join(path, 'test_batch')
    x_test, y_test = load_batch(fpath)
  
    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))
  
    if K.image_data_format() == 'channels_last':
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)
    return (x_train, y_train), (x_test, y_test)

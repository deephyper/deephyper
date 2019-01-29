import os
from keras.utils.data_utils import get_file

origin = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
here = os.getcwd()
here = os.path.abspath(here)
dest = os.path.join(here, 'cifar-10-python.tar.gz')
path = get_file(fname=dest, origin=origin)

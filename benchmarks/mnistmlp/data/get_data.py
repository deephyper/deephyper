import os
from keras.utils.data_utils import get_file

origin='https://s3.amazonaws.com/img-datasets/mnist.npz'
file_hash='8a61469f7ea1b51cbae51d4f78837e45'

here = os.getcwd()
here = os.path.abspath(here)
dest = os.path.join(here, 'mnist.npz')

path = get_file(fname=dest, origin=origin)

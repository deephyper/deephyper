import tensorflow as tf
import numpy as np
from deephyper.search.nas.model.space.layers import Padding
from pprint import pprint

def test_padding_layer():
    model = tf.keras.Sequential()
    model.add(Padding([[1, 1]]))

    data = np.random.random((3, 1))
    shape_data = np.shape(data)
    assert shape_data == (3, 1)

    res = model.predict(data, batch_size=1)
    res_shape = np.shape(res)
    assert res_shape == (3, 3)

if __name__ == '__main__':
    test_padding_layer()
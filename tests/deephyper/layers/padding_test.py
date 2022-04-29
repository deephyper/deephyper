import pytest


@pytest.mark.nas
def test_padding_layer():
    import tensorflow as tf
    import numpy as np
    from deephyper.layers import Padding

    model = tf.keras.Sequential()
    model.add(Padding([[1, 1]]))

    data = np.random.random((3, 1))
    shape_data = np.shape(data)
    assert shape_data == (3, 1)

    res = model.predict(data, batch_size=1)
    res_shape = np.shape(res)
    assert res_shape == (3, 3)

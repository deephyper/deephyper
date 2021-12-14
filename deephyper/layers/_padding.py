import tensorflow as tf


class Padding(tf.keras.layers.Layer):

    """Multi-dimensions padding layer.

    This operation pads a tensor according to the paddings you specify. paddings is an
    integer tensor with shape [n-1, 2], where n is the rank of tensor. For each dimension
    D of input, paddings[D, 0] indicates how many values to add before the contents of
    tensor in that dimension, and paddings[D, 1] indicates how many values to add after
    the contents of tensor in that dimension. The first dimension corresponding to the
    batch size cannot be padded.

    Args:
        padding (list(list(int))): e.g. [[1, 1]]
        mode (str): 'CONSTANT', 'REFLECT' or 'SYMMETRIC'

    """

    def __init__(self, padding, mode="CONSTANT", constant_values=0, **kwargs):
        super(Padding, self).__init__(**kwargs)
        self.padding = [[0, 0]] + padding
        self.mode = mode
        self.constant_values = constant_values

    def call(self, x, mask=None):
        padding = tf.constant(self.padding)
        return tf.pad(
            tensor=x,
            paddings=padding,
            mode=self.mode,
            constant_values=self.constant_values,
        )

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(
            [
                input_shape[i] + sum(self.padding[i])
                if not input_shape[i] is None
                else None
                for i in range(len(input_shape))
            ]
        )

    def get_config(self):
        config = {
            "padding": self.padding[1:],
            "mode": self.mode,
            "constant_values": self.constant_values,
        }
        base_config = super(Padding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

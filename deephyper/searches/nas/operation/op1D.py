import tensorflow as tf

from deephyper.searches.nas.operation.basic import Operation


class Conv1D(Operation):
    """Convolution for one dimension.

    Help you to create a one dimension convolution operation.

    Args:
        filter_size (int): size kernels/filters
        num_filters (int): number of kernels/filters
        strides (int):
        padding (str): 'SAME' or 'VALID'
    """
    def __init__(self, filter_size, num_filters=1, strides=1, padding='SAME'):
        self.filter_size = filter_size
        self.num_filters = num_filters
        self.strides = strides
        self.padding = padding

    def __str__(self):
        return f'Conv1D_{self.filter_size}_{self.num_filters}'

    def __call__(self, inputs, **kwargs):
        inpt = inputs[0]
        out = tf.layers.conv1d(inpt,
            filters=self.num_filters,
            kernel_size=self.filter_size,
            strides=self.strides,
            padding=self.padding,
            kernel_initializer=tf.initializers.random_uniform(),
            activation=None,
        )
        return out

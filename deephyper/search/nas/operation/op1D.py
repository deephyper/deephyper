import tensorflow as tf

from deephyper.search.nas.operation.basic import Operation


class Conv1D(Operation):
    def __init__(self, filter_size, num_filters=1, stride=1, padding='SAME'):
        self.filter_size = filter_size
        self.num_filters = num_filters
        self.stride = stride
        self.padding = padding

    def __str__(self):
        return f'Conv1D_{self.filter_size}_{self.num_filters}'

    def __call__(self, inputs, **kwargs):
        inpt = tf.expand_dims(inputs[0], -1)
        kernel = tf.get_variable('kernel_conv1D', shape=(self.filter_size, 1, self.num_filters), dtype=tf.float32, initializer=tf.initializers.random_uniform())
        out = tf.contrib.layers.flatten(tf.nn.conv1d(inpt, kernel, self.stride, self.padding))
        return out

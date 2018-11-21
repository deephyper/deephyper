import tensorflow as tf
from tensorflow import keras
import numpy as np

from deephyper.search.nas.model.space.op.basic import Operation

class Concatenate(Operation):
    """Concatenate operation.

    Args:
        graph:
        node (Node): current_node of the operation
        stacked_nodes (list(Node)): nodes to concatenate
        axis (int): axis to concatenate
    """
    def __init__(self, graph=None, node=None, stacked_nodes=None, axis=-1):
        self.graph = graph
        self.node = node
        self.stacked_nodes = stacked_nodes
        self.axis = axis

    def is_set(self):
        if self.stacked_nodes is not None:
            for n in self.stacked_nodes:
                self.graph.add_edge(n, self.node)

    def __call__(self, values, **kwargs):
        if len(values) > 1:
            out = keras.layers.Concatenate(axis=-1)(values)
        else:
            out = values[0]
        return out

class Dense(Operation):
    """Multi Layer Perceptron operation.

    Help you to create a perceptron with n layers, m units per layer and an activation function.

    Args:
        layers (int): number of layers.
        units (int): number of units per layer.
        activation: an activation function from tensorflow.
    """
    def __init__(self, units, activation):
        self.units = units
        self.activation = activation

    def __str__(self):
        return f'Dense_{self.units}_{self.activation.__name__}'

    def __call__(self, inputs, **kwargs):
        assert len(inputs) == 1, f'{type(self).__name__} as {len(inputs)} inputs when 1 is required.'
        out = keras.layers.Dense(
            units=self.units,
            activation=self.activation,
            kernel_initializer=tf.initializers.random_uniform())(inputs[0])
        return out


class Dropout(Operation):
    """Dropout operation.

    Help you to create a dropout operation.

    Args:
        rate (float): rate of deactivated inputs.
    """
    def __init__(self, rate):
        self.rate = rate

    def __str__(self):
        return f'Dropout({int((1.-self.rate)*100)})'

    def __call__(self, inputs, **kwargs):
        assert len(inputs) == 1, f'{type(self).__name__} as {len(inputs)} inputs when 1 is required.'
        inpt = inputs[0]
        out = keras.layers.Dropout(rate=self.rate)(inpt)
        return out


dropout_ops = [Dropout(0.),
               Dropout(0.1),
               Dropout(0.2),
               Dropout(0.3),
               Dropout(0.4),
               Dropout(0.5),
               Dropout(0.6)]

class Identity(Operation):
    def __call__(self, inputs, **kwargs):
        assert len(inputs) == 1, f'{type(self).__name__} as {len(inputs)} inputs when 1 is required.'
        return inputs[0]


class Conv1D(Operation):
    """Convolution for one dimension.

    Help you to create a one dimension convolution operation.

    Args:
        filter_size (int): size kernels/filters
        num_filters (int): number of kernels/filters
        strides (int):
        padding (str): 'same' or 'valid'
    """
    def __init__(self, filter_size, num_filters=1, strides=1, padding='same'):
        self.filter_size = filter_size
        self.num_filters = num_filters
        self.strides = strides
        self.padding = padding

    def __str__(self):
        return f'{type(self).__name__}_{self.filter_size}_{self.num_filters}'

    def __call__(self, inputs, **kwargs):
        assert len(inputs) == 1, f'{type(self).__name__} as {len(inputs)} inputs when only 1 is required.'
        inpt = inputs[0]
        out = keras.layers.Conv1D(filters=self.num_filters, kernel_size=self.filter_size, strides=self.strides, padding=self.padding)(inpt)
        return out


class MaxPooling1D(Operation):
    """MaxPooling over one dimension.

    Args:
        pool_size ([type]): [description]
        strides (int, optional): Defaults to 1. [description]
        padding (str, optional): Defaults to 'valid'. [description]
        data_format (str, optional): Defaults to 'channels_last'. [description]
    """
    def __init__(self, pool_size, strides=1, padding='valid', data_format='channels_last'):
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding
        self.data_format = data_format

    def __str__(self):
        return f'{type(self).__name__}_{self.pool_size}_{self.padding}'

    def __call__(self, inputs, **kwargs):
        assert len(inputs) == 1, f'{type(self).__name__} as {len(inputs)} inputs when only 1 is required.'
        out = keras.layers.MaxPooling1D(
            pool_size=self.pool_size,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format
        )
        return out

class Flatten(Operation):
    """Flatten operation.

    Args:
        data_format (str, optional): Defaults to None.
    """
    def __init__(self, data_format=None):
        self.data_format = data_format

    def __call__(self, inputs, **kwargs):
        assert len(inputs) == 1, f'{type(self).__name__} as {len(inputs)} inputs when only 1 is required.'
        out = keras.layers.Flatten(
            data_format=self.data_format
        )
        return out

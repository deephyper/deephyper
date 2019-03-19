import tensorflow as tf
from tensorflow import keras
import numpy as np

from deephyper.search.nas.model.space.op.basic import Operation
import deephyper.search.nas.model.space.layers as deeplayers

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
        len_shp = max([len(x.get_shape()) for x in values])

        if len_shp > 3:
            raise RuntimeError('This concatenation is for 2D or 3D tensors only when a {len_shp}D is passed!')

        # zeros padding
        if len(values) > 1:


            if all(map(lambda x: len(x.get_shape())==len_shp or \
                len(x.get_shape())==(len_shp-1), values)): # all tensors should have same number of dimensions 2d or 3d, but we can also accept a mix of 2d en 3d tensors
                for i, v in enumerate(values): # we have a mix of 2d and 3d tensors so we are expanding 2d tensors to be 3d with last_dim==1
                    if len(v.get_shape()) < len_shp:
                        values[i] = keras.layers.Reshape((*tuple(v.get_shape()[1:]), 1))(v)
                if len_shp == 3: # for 3d tensors concatenation is applied along last dim (axis=-1), so we are applying a zero padding to make 2nd dimensions (ie. shape()[1]) equals
                    max_len = max(map(lambda x: int(x.get_shape()[1]), values))
                    paddings = map(lambda x: max_len - int(x.get_shape()[1]), values)
                    for i, (p, v) in enumerate(zip(paddings, values)):
                        lp = p // 2
                        rp = p - lp
                        values[i] = keras.layers.ZeroPadding1D(padding=(lp, rp))(v)
                # elif len_shp == 2 nothing to do
            else:
                raise RuntimeError(
                    f'All inputs of concatenation operation should have same shape length:\n'
                    f'number_of_inputs=={len(values)}\n'
                    f'shape_of_inputs=={[str(x.get_shape()) for x in values]}')

        # concatenation
        if len(values) > 1:
            out = keras.layers.Concatenate(axis=-1)(values)
        else:
            out = values[0]
        return out



class Dense(Operation):
    """Multi Layer Perceptron operation.

    Help you to create a perceptron with n layers, m units per layer and an activation function.

    Args:
        units (int): number of units per layer.
        activation: an activation function from tensorflow.
    """
    def __init__(self, units, activation=None, *args, **kwargs):
        # Layer args
        self.units = units
        self.activation = activation
        self.kwargs = kwargs

        # Reuse arg
        self._layer = None

    def __str__(self):
        if isinstance(self.activation, str):
            return f'Dense_{self.units}_{self.activation}'
        elif self.activation is None:
            return f'Dense_{self.units}'
        else:
            return f'Dense_{self.units}_{self.activation.__name__}'

    def __call__(self, inputs, *args, **kwargs):
        assert len(inputs) == 1, f'{type(self).__name__} as {len(inputs)} inputs when 1 is required.'
        if self._layer is None: # reuse mechanism
            self._layer = keras.layers.Dense(
            units=self.units,
            activation=self.activation,
            **self.kwargs,
            )
        out = self._layer(inputs[0])
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
        return f'Dropout({self.rate})'

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
        self._layer = None

    def __str__(self):
        return f'{type(self).__name__}_{self.filter_size}_{self.num_filters}'

    def __call__(self, inputs, **kwargs):
        assert len(inputs) == 1, f'{type(self).__name__} as {len(inputs)} inputs when only 1 is required.'
        inpt = inputs[0]
        if len(inpt.get_shape()) == 2:
            out = keras.layers.Reshape((inpt.get_shape()[1], 1))(inpt)
        else:
            out = inpt
        if self._layer is None: # reuse mechanism
            self._layer = keras.layers.Conv1D(
            filters=self.num_filters,
            kernel_size=self.filter_size,
            strides=self.strides,
            padding=self.padding)
        out = self._layer(out)
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
        inpt = inputs[0]
        if len(inpt.get_shape()) == 2:
            out = keras.layers.Reshape((inpt.get_shape()[1], 1))(inpt)
        else:
            out = inpt
        out = keras.layers.MaxPooling1D(
            pool_size=self.pool_size,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format
        )(out)
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
        inpt = inputs[0]
        if len(inpt.get_shape()) == 2:
            out = inpt
        else:
            out = keras.layers.Flatten(
                data_format=self.data_format
            )(inpt)
        return out

class Activation(Operation):
    """Activation function operation.

    Args:
        activation (callable): an activation function
    """
    def __init__(self, activation=None, *args, **kwargs):
        self.activation = activation
        self._layer = None

    def __str__(self):
        return f'{type(self).__name__}_{self.activation}'

    def __call__(self, inputs, *args, **kwargs):
        inpt = inputs[0]
        if self._layer is None:
            self._layer = keras.layers.Activation(activation=self.activation)
        out = self._layer(inpt)
        return out

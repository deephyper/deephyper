import tensorflow as tf
from tensorflow import keras

from . import Operation

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

    def __call__(self, inputs, seed=None, **kwargs):
        assert len(
            inputs) == 1, f'{type(self).__name__} as {len(inputs)} inputs when 1 is required.'
        if self._layer is None:  # reuse mechanism
            self._layer = keras.layers.Dense(
                units=self.units,
                kernel_initializer=tf.keras.initializers.glorot_uniform(seed=seed),
                **self.kwargs,
            )

        out = self._layer(inputs[0])
        if self.activation is not None: # better for visualisation
            out = keras.layers.Activation(activation=self.activation)(out)
        return out


class Dropout(Operation):
    """Dropout operation.

    Help you to create a dropout operation.

    Args:
        rate (float): rate of deactivated inputs.
    """

    def __init__(self, rate):
        self.rate = rate
        super().__init__(layer=keras.layers.Dropout(rate=self.rate))

    def __str__(self):
        return f'Dropout({self.rate})'

class Identity(Operation):
    def __init__(self):
        pass

    def __call__(self, inputs, **kwargs):
        assert len(
            inputs) == 1, f'{type(self).__name__} as {len(inputs)} inputs when 1 is required.'
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
        assert len(
            inputs) == 1, f'{type(self).__name__} as {len(inputs)} inputs when only 1 is required.'
        inpt = inputs[0]
        if len(inpt.get_shape()) == 2:
            out = keras.layers.Reshape((inpt.get_shape()[1], 1))(inpt)
        else:
            out = inpt
        if self._layer is None:  # reuse mechanism
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
        assert len(
            inputs) == 1, f'{type(self).__name__} as {len(inputs)} inputs when only 1 is required.'
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
        assert len(
            inputs) == 1, f'{type(self).__name__} as {len(inputs)} inputs when only 1 is required.'
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

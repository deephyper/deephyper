import tensorflow as tf

from . import Operation
from deephyper.contrib.layers.timestep_dropout import (
    TimestepDropout as TimestepDropoutLayer,
)


class LSTM(Operation):
    def __init__(self, units, activation=None, return_sequences=False, *args, **kwargs):
        # Layer args
        self.units = units
        self.activation = activation
        self.return_sequences = return_sequences
        self.kwargs = kwargs

        # Reuse arg
        self._layer = None

    def __str__(self):
        if isinstance(self.activation, str):
            return f"LSTM_{self.units}_{self.activation}"
        elif self.activation is None:
            return f"LSTM_{self.units}"
        else:
            return f"LSTM_{self.units}_{self.activation.__name__}"

    def __call__(self, inputs, seed=None, **kwargs):
        assert (
            len(inputs) == 1
        ), f"{type(self).__name__} as {len(inputs)} inputs when 1 is required."
        if self._layer is None:  # reuse mechanism
            self._layer = tf.keras.layers.LSTM(
                units=self.units,
                activation=self.activation,
                return_sequences=self.return_sequences,
                kernel_initializer=tf.keras.initializers.glorot_uniform(seed=seed),
                **self.kwargs,
            )

        out = self._layer(inputs[0])
        return out


class Embedding(Operation):
    def __init__(self, input_dim, output_dim, *args, **kwargs):
        # Layer args
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kwargs = kwargs

        # Reuse arg
        self._layer = None

    def __str__(self):
        return f"Embedding_{self.input_dim}_{self.output_dim}"

    def __call__(self, inputs, seed=None, **kwargs):
        assert (
            len(inputs) == 1
        ), f"{type(self).__name__} as {len(inputs)} inputs when 1 is required."
        if self._layer is None:  # reuse mechanism
            self._layer = tf.keras.layers.Embedding(
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                embeddings_initializer=tf.keras.initializers.glorot_uniform(seed=seed),
                **self.kwargs,
            )

        out = self._layer(inputs[0])
        return out


class TimestepDropout(Operation):
    def __init__(self, rate):
        # Layer args
        self.rate = rate

        # Reuse arg
        self._layer = None

    def __str__(self):
        return f"TimestepDropout_{self.rate}"

    def __call__(self, inputs, seed=None, **kwargs):
        assert (
            len(inputs) == 1
        ), f"{type(self).__name__} as {len(inputs)} inputs when 1 is required."
        if self._layer is None:  # reuse mechanism
            self._layer = TimestepDropoutLayer(rate=self.rate)

        out = self._layer(inputs[0])
        return out

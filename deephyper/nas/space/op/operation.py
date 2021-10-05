import tensorflow as tf


class Operation:
    """Interface of an operation.

    >>> import tensorflow as tf
    >>> from deephyper.nas.space.op import Operation
    >>> Operation(layer=tf.keras.layers.Dense(10))
    Dense

    Args:
        layer (Layer): a ``tensorflow.keras.layers.Layer``.
    """

    def __init__(self, layer: tf.keras.layers.Layer):
        assert isinstance(layer, tf.keras.layers.Layer)
        self.from_keras_layer = True
        self._layer = layer

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        if hasattr(self, "from_keras_layer"):
            return type(self._layer).__name__
        else:
            try:
                return str(self)
            except:
                return type(self).__name__

    def __call__(self, tensors: list, seed: int = None, **kwargs):
        """
        Args:
            tensors (list): a list of incoming tensors.

        Returns:
            tensor: an output tensor.
        """
        if len(tensors) == 1:
            out = self._layer(tensors[0])
        else:
            out = self._layer(tensors)
        return out

    def init(self, current_node):
        """Preprocess the current operation."""


def operation(cls):
    """Dynamically create a sub-class of Operation from a Keras layer."""

    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs
        self._layer = None

    def __repr__(self):
        return cls.__name__

    def __call__(self, inputs, **kwargs):

        if self._layer is None:
            self._layer = cls(*self._args, **self._kwargs)

        if len(inputs) == 1:
            out = self._layer(inputs[0])
        else:
            out = self._layer(inputs)
        return out

    cls_attrs = dict(__init__=__init__, __repr__=__repr__, __call__=__call__)
    op_class = type(cls.__name__, (Operation,), cls_attrs)

    return op_class
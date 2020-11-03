from tensorflow import keras


class Operation:
    """Interface of an operation.

    >>> import tensorflow as tf
    >>> from deephyper.nas.space.op.op1d import Operation
    >>> Operation(layer=tf.keras.layers.Dense(10))
    Dense

    Args:
        layer (Layer): a ``tensorflow.keras.layers.Layer``.
    """

    def __init__(self, layer: keras.layers.Layer):
        assert isinstance(layer, keras.layers.Layer)
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
        """Preprocess the current operation.
        """


class Tensor(Operation):
    def __init__(self, tensor, *args, **kwargs):
        self.tensor = tensor

    def __str__(self):
        return str(self.tensor)

    def __call__(self, *args, **kwargs):
        return self.tensor


class Zero(Operation):
    def __init__(self):
        self.tensor = []

    def __str__(self):
        return "Zero"

    def __call__(self, *args, **kwargs):
        return self.tensor

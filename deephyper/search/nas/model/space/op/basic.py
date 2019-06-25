import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Layer

import deephyper.search.nas.model.space.layers as deeplayers


class Operation:
    """Interface of an operation.

    Args:
        layer (Layer): a ``tensorflow.keras`` layer.
    """

    def __init__(self, layer: Layer):
        assert isinstance(layer, Layer)
        self._layer = layer

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        if hasattr(self, '_layer'):
            return type(self._layer).__name__
        else:
            return type(self).__name__

    def __call__(self, tensors: list, *args, **kwargs):
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

    def init(self):
        """Preprocess the current operation.
        """


class Tensor(Operation):
    def __init__(self, tensor, *args, **kwargs):
        self.tensor = tensor

    def __call__(self, *args, **kwargs):
        return self.tensor

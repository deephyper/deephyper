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
            except Exception:
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
    """Dynamically creates a sub-class of Operation from a Keras layer.

    Args:
        cls (tf.keras.layers.Layer): takes a Keras layer class as input and return an operation class corresponding to this layer.
    """

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


class Identity(Operation):
    def __init__(self):
        pass

    def __call__(self, inputs, **kwargs):
        assert (
            len(inputs) == 1
        ), f"{type(self).__name__} as {len(inputs)} inputs when 1 is required."
        return inputs[0]


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


class Connect(Operation):
    """Connection node.

    Represents a possibility to create a connection between n1 -> n2.

    Args:
        graph (nx.DiGraph): a graph
        source_node (Node): source
    """

    def __init__(self, search_space, source_node, *args, **kwargs):
        self.search_space = search_space
        self.source_node = source_node
        self.destin_node = None

    def __str__(self):
        if type(self.source_node) is list:
            if len(self.source_node) > 0:
                ids = str(self.source_node[0].id)
                for n in self.source_node[1:]:
                    ids += "," + str(n.id)
            else:
                ids = "None"
        else:
            ids = self.source_node.id
        if self.destin_node is None:
            return f"{type(self).__name__}_{ids}->?"
        else:
            return f"{type(self).__name__}_{ids}->{self.destin_node.id}"

    def init(self, current_node):
        """Set the connection in the search_space graph from n1 -> n2."""
        self.destin_node = current_node
        if type(self.source_node) is list:
            for n in self.source_node:
                self.search_space.connect(n, self.destin_node)
        else:
            self.search_space.connect(self.source_node, self.destin_node)

    def __call__(self, value, *args, **kwargs):
        return value

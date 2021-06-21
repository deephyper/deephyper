from deephyper.nas.space.op.basic import Operation

__all__ = ["Operation", "operation"]


def operation(cls):
    """Dynamically create a sub-class of Operation from a Keras layer.
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
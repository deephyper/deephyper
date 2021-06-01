import json
import types
import uuid

import skopt
from numpy import bool_, floating, integer, ndarray
import ConfigSpace.hyperparameters as csh


class Encoder(json.JSONEncoder):
    """
    Enables JSON dump of numpy data, python functions.
    """

    def default(self, obj):
        if isinstance(obj, uuid.UUID):
            return str(obj)
        elif isinstance(obj, integer):
            return int(obj)
        elif isinstance(obj, floating):
            return float(obj)
        elif isinstance(obj, bool_):
            return bool(obj)
        elif isinstance(obj, ndarray):
            return obj.tolist()
        elif isinstance(obj, types.FunctionType):
            return f"{obj.__module__}.{obj.__name__}"
        elif isinstance(obj, skopt.space.Dimension):
            return str(obj)
        elif isinstance(obj, csh.Hyperparameter):
            return str(obj)
        else:
            return super(Encoder, self).default(obj)
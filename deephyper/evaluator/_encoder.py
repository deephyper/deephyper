import json
import types
import uuid
from inspect import isclass

import ConfigSpace as cs
import ConfigSpace.hyperparameters as csh
import deephyper.skopt
import numpy as np
from ConfigSpace.read_and_write import json as cs_json


class Encoder(json.JSONEncoder):
    """
    Enables JSON dump of numpy data, python functions.
    """

    def default(self, obj):
        if isinstance(obj, uuid.UUID):
            return str(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, types.FunctionType) or isclass(obj):
            return f"{obj.__module__}.{obj.__name__}"
        elif isinstance(obj, deephyper.skopt.space.Dimension):
            return str(obj)
        elif isinstance(obj, csh.Hyperparameter):
            return str(obj)
        elif isinstance(obj, cs.ConfigurationSpace):
            return json.loads(cs_json.write(obj))
        else:
            return super(Encoder, self).default(obj)

import json
import re
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


def to_json(d: dict):
    return json.dumps(d, cls=Encoder)


def parse_subprocess_result(result):
    """Utility to parse a result from a subprocess of the format `"DH-OUTPUT:..."`.

    Args:
        result: object returned by a subpross with ``stdout`` and ``stderr`` attributes.

    Return:
        The parsed value or raise an exception if an error happened.
    """
    stdout = result.stdout
    stderr = result.stderr
    try:
        retval_bytes = re.search(b"DH-OUTPUT:(.+)\n", stdout).group(1)
    except AttributeError:
        error = stderr.decode("utf-8")
        raise RuntimeError(
            f"{error}\n\n Could not collect any result from the run_function in the main process because an error happened in the subprocess."
        )
    # Finally, parse whether the return value from the user-defined function is a scalar, a list, or a dictionary.
    retval = retval_bytes.replace(
        b"'", b'"'
    )  # For dictionaries, replace single quotes with double quotes!
    sol = json.loads(retval)
    return sol

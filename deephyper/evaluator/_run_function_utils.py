from typing import Union
from numbers import Number

import numpy as np


def standardize_run_function_output(
    output: Union[str, float, tuple, list, dict],
) -> dict:
    """Transform the output of the run-function to its standard form.

    Possible return values of the run-function are:

    >>> 0
    >>> 0, 0
    >>> "F_something"
    >>> {"objective": 0 }
    >>> {"objective": (0, 0), "metadata": {...}}

    Args:
        output (Union[str, float, tuple, list, dict]): the output of the run-function.

    Returns:
        dict: standardized output of the function.
    """

    # output returned a single objective value
    if np.isscalar(output):
        if isinstance(output, str):
            output = {"objective": output}
        elif isinstance(output, Number):
            output = {"objective": float(output)}
        else:
            raise TypeError(
                f"The output of the run-function cannot be of type {type(output)} it should be either a string or a number."
            )

    # output only returned objective values as tuple or list
    elif isinstance(output, (tuple, list)):

        output = {"objective": output}

    elif isinstance(output, dict):
        pass
    else:
        raise TypeError(
            f"The output of the run-function cannot be of type {type(output)}"
        )

    metadata = output.get("metadata", dict())
    if metadata is None:
        metadata = dict()
    elif not isinstance(metadata, dict):
        raise TypeError(
            f"The metadata of the run-function cannot be of type {type(metadata)}"
        )
    output["metadata"] = metadata

    # check if multiple observations returned
    objective = np.asarray(output["objective"])
    if objective.ndim == 2:
        output["objective"] = objective[1, -1].tolist()
        output["observations"] = objective.tolist()

    return output

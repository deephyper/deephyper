import inspect
import json


def _get_init_param_names(cls):
    """Get parameter names for the estimator"""
    # fetch the constructor
    init = cls.__init__
    if init is object.__init__:
        # No explicit constructor to introspect
        return []

    # introspect the constructor arguments to find the model parameters
    # to represent
    init_signature = inspect.signature(init)
    # Consider the constructor parameters excluding 'self'
    parameters = [
        p
        for p in init_signature.parameters.values()
        if p.kind == p.POSITIONAL_OR_KEYWORD and p.name not in ["self"]
    ]
    # Extract and sort argument names excluding 'self'
    return sorted([p.name for p in parameters])


def get_init_params(obj):
    """Get the raw parameters of an object.

    Args:
        obj (any): The object of which we want to know the ``__init__`` arguments.

    Returns:
        params (dict): Parameter names mapped to their values.
    """
    if hasattr(obj, "_init_params"):
        base_init_params = obj._init_params
    else:
        base_init_params = dict()
    params = dict()
    for key in _get_init_param_names(obj):
        if hasattr(obj, f"_{key}"):
            value = getattr(obj, f"_{key}")
        elif hasattr(obj, f"{key}"):
            value = getattr(obj, f"{key}")
        else:
            value = base_init_params.get(key, None)
        params[key] = value
    return params


def get_init_params_as_json(obj):
    """Get the parameters of an object in a json format.

    Args:
        obj (any): The object of which we want to know the ``__init__`` arguments.

    Returns:
        params (dict): Parameter names mapped to their values.
    """
    if hasattr(obj, "_init_params"):
        base_init_params = obj._init_params
    else:
        base_init_params = dict()
    params = dict()
    for key in _get_init_param_names(obj):
        if hasattr(obj, f"_{key}"):
            value = getattr(obj, f"_{key}")
        elif hasattr(obj, f"{key}"):
            value = getattr(obj, f"{key}")
        else:
            value = base_init_params.get(key, None)
        if hasattr(value, "to_json"):
            value = value.to_json()
        try:
            json.dumps(value)
        except:
            value = type(value).__name__
        params[key] = value
    return params

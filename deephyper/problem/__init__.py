import ConfigSpace as config_space
from inspect import signature
from .hyperparameter import HpProblem
from .neuralarchitecture import NaProblem

__all__ = ["config_space", "HpProblem", "NaProblem", "filter_parameters"]


def filter_parameters(obj, config: dict) -> dict:
    """Filter the incoming configuration dict based on the signature of obj.

    Args:
        obj (Callable): the object for which the signature is used.
        config (dict): the configuration to filter.

    Returns:
        dict: the filtered configuration dict.
    """
    sig = signature(obj)
    clf_allowed_params = list(sig.parameters.keys())
    clf_params = {
        k: v
        for k, v in config.items()
        if k in clf_allowed_params and not (v in ["nan", "NA"])
    }
    return clf_params

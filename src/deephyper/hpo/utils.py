"""Utilities from search algorithms."""

from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    Constant,
    NumericalHyperparameter,
    OrdinalHyperparameter,
)

__all__ = [
    "get_inactive_value_of_hyperparameter",
]


def get_inactive_value_of_hyperparameter(hp):
    """Return the value of an hyperparameter when considered inactive."""
    if isinstance(hp, NumericalHyperparameter):
        return hp.lower
    elif isinstance(hp, CategoricalHyperparameter):
        return hp.choices[0]
    elif isinstance(hp, OrdinalHyperparameter):
        return hp.sequence[0]
    elif isinstance(hp, Constant):
        return hp.value
    else:
        raise ValueError(f"Unsupported hyperparameter type: {type(hp)}")

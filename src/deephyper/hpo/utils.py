"""Utilities from search algorithms."""

import numpy as np
import pandas as pd
from typing import Tuple
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


def get_mask_of_rows_without_failures(df: pd.DataFrame, column: str) -> Tuple[bool, np.ndarray]:
    """Return a boolean mask where true values are non-failures.

    Returns:
        bool, Array[bool]: a boolean that indicates if there is any failure, the mask array.
    """
    if pd.api.types.is_string_dtype(df[column]):
        mask_no_failures = ~df[column].str.startswith("F").values
    else:
        mask_no_failures = df[column].map(lambda x: isinstance(x, float)).values
    has_any_failure = not np.all(mask_no_failures)
    return has_any_failure, mask_no_failures

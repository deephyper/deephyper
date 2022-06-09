"""The preprocessing module provides a few functions which returns a preprocessing pipeline following the Scikit-Learn API.
"""
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def stdscaler() -> Pipeline:
    """Standard normalization where the mean is of each row is set to zero and the standard deviation is set to one.

    Returns:
        Pipeline: a pipeline with one step ``StandardScaler``.
    """
    preprocessor = Pipeline([("stdscaler", StandardScaler())])
    return preprocessor


def minmaxscaler() -> Pipeline:
    """Standard normalization where the mean is of each row is set to zero and the standard deviation is set to one.

    Returns:
        Pipeline: a pipeline with one step ``StandardScaler``.
    """
    preprocessor = Pipeline([("minmaxscaler", MinMaxScaler())])
    return preprocessor


def minmaxstdscaler() -> Pipeline:
    """MinMax preprocesssing followed by Standard normalization.

    Returns:
        Pipeline: a pipeline with two steps ``[MinMaxScaler, StandardScaler]``.
    """
    preprocessor = Pipeline(
        [
            ("minmaxscaler", MinMaxScaler()),
            ("stdscaler", StandardScaler()),
        ]
    )
    return preprocessor

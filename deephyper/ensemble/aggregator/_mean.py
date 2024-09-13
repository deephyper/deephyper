from typing import List

import numpy as np

from deephyper.ensemble.aggregator._aggregator import Aggregator


def average(x: np.ndarray | np.ma.MaskedArray, axis=None, weights=None):
    """Check if ``x`` is a classic numpy array or a masked array to apply the corresponding
    implementation.

    Args:
        x (np.ndarray | np.ma.MaskedArray): array like.
        axis (_type_, optional): the axis. Defaults to ``None``.
        weights (_type_, optional): the weights. Defaults to ``None``.

    Returns:
        array like: the average.
    """
    numpy_func = np.average
    if isinstance(x, np.ma.MaskedArray):
        numpy_func = np.ma.average
    return numpy_func(x, axis=axis, weights=weights)


class MeanAggregator(Aggregator):
    """Aggregate the predictions using the average."""

    def aggregate(self, y: List[np.ndarray | np.ma.MaskedArray], weights: List = None):
        """Aggregate the predictions using the mean.

        Args:
            y (np.array): Predictions array of shape ``(n_predictors, n_samples, n_outputs)``.

        Returns:
            np.array: Aggregated predictions of shape ``(n_samples, n_outputs)``.
        """
        return average(y, axis=0, weights=weights)

from typing import List

import numpy as np

from deephyper.ensemble.aggregator._aggregator import Aggregator


class MeanAggregator(Aggregator):
    """Aggregate the predictions using the average."""

    def aggregate(self, y: List[np.ndarray | np.ma.MaskedArray], weights: List = None):
        """Aggregate the predictions using the mean.

        Args:
            y (np.array): Predictions array of shape ``(n_predictors, n_samples, n_outputs)``.

        Returns:
            np.array: Aggregated predictions of shape ``(n_samples, n_outputs)``.
        """
        if isinstance(y, np.ma.MaskedArray):
            y = np.ma.average(y, axis=0, weights=weights)
        else:
            y = np.average(y, axis=0, weights=weights)
        return y

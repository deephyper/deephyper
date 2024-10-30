from typing import List

import numpy as np

from deephyper.ensemble.aggregator._aggregator import Aggregator
from deephyper.ensemble.aggregator.utils import average


class MeanAggregator(Aggregator):
    """Aggregate the predictions using the average.

    .. list-table::
        :widths: 25 25
        :header-rows: 1

        * - Array (Fixed Set)
          - MaskedArray
        * - ✅
          - ✅

    """

    def aggregate(self, y: List[np.ndarray | np.ma.MaskedArray], weights: List = None):
        """Aggregate the predictions using the mean.

        Args:
            y (np.array): Predictions array of shape ``(n_predictors, n_samples, n_outputs)``.

        Returns:
            np.array: Aggregated predictions of shape ``(n_samples, n_outputs)``.
        """
        return average(y, axis=0, weights=weights)

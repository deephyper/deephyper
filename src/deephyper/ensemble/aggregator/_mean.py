from typing import List, Optional, Union

import numpy as np

from deephyper.ensemble.aggregator._aggregator import Aggregator


class MeanAggregator(Aggregator):
    """Aggregate predictions using the mean. Supports both NumPy arrays and masked arrays.

    .. list-table::
        :widths: 25 25
        :header-rows: 1

        * - Array (Fixed Set)
          - MaskedArray
        * - ✅
          - ✅

    Args:
        with_uncertainty (bool, optional): a boolean that sets if the uncertainty should be returned
            when calling the aggregator. Defaults to ``False``.
    """

    def __init__(self, with_uncertainty: bool = False):
        self.with_uncertainty = with_uncertainty

    def aggregate(
        self,
        y: List[Union[np.ndarray, np.ma.MaskedArray]],
        weights: Optional[List[float]] = None,
    ) -> Union[np.ndarray, np.ma.MaskedArray]:
        """Aggregate predictions using the mean.

        Args:
            y (List[np.ndarray | np.ma.MaskedArray]): List of prediction arrays, each of shape
                ``(n_samples, n_outputs)``.

            weights (Optional[List[float]]): Optional weights for the predictors. If provided,
                must have the same length as `y`.

        Returns:
            np.ndarray: Aggregated predictions of shape ``(n_samples, n_outputs)``.

        Raises:
            ValueError: If `weights` length does not match the number of predictors in `y`.
        """
        if weights is not None and len(weights) != len(y):
            raise ValueError("The length of `weights` must match the number of predictors in `y`.")

        # Ensure `y` is a valid list of arrays
        if not all(isinstance(pred, (np.ndarray, np.ma.MaskedArray)) for pred in y):
            raise TypeError("All elements of `y` must be numpy.ndarray or numpy.ma.MaskedArray.")

        self._np = np
        if all(isinstance(pred, np.ma.MaskedArray) for pred in y):
            self._np = np.ma

        # Stack predictions for aggregation
        stacked_y = self._np.stack(y, axis=0)
        avg = self._np.average(stacked_y, axis=0, weights=weights)

        if not self.with_uncertainty:
            return avg

        uncertainty = self._np.average(self._np.square(stacked_y - avg), axis=0, weights=weights)

        return {"loc": avg, "uncertainty": uncertainty}

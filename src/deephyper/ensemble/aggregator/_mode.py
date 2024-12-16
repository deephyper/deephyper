from typing import List, Optional, Union, Dict

import numpy as np

from deephyper.ensemble.aggregator._aggregator import Aggregator


class ModeAggregator(Aggregator):
    """Aggregate predictions using the mode of categorical distributions from predictors.

    .. list-table::
        :widths: 25 25
        :header-rows: 1

        * - Array (Fixed Set)
          - MaskedArray
        * - ✅
          - ✅

    This aggregator is useful when the ensemble is composed of predictors that output categorical
    distributions. The mode of the ensemble is the mode of the modes of the predictors, minimizing
    the 0-1 loss.

    Args:
        with_uncertainty (bool, optional): a boolean that sets if the uncertainty should be
        returned when calling the aggregator. Defaults to ``False``.
    """

    def __init__(self, with_uncertainty: bool = False):
        self.with_uncertainty = with_uncertainty

    def aggregate(
        self,
        y: List[Union[np.ndarray, np.ma.MaskedArray]],
        weights: Optional[List[float]] = None,
    ) -> Union[
        Union[np.ndarray, np.ma.MaskedArray],
        Dict[str, Union[np.ndarray, np.ma.MaskedArray]],
    ]:
        """Aggregate predictions using the mode of categorical distributions.

        Args:
            y (List[Union[np.ndarray, np.ma.MaskedArray]]): List of categorical probability arrays
                of shape ``(n_predictors, n_samples, ..., n_classes)``.
            weights (Optional[List[float]]): Weights for the predictors. Default is ``None``.

        Returns:
            Union[Union[np.ndarray, np.ma.MaskedArray], Dict[str, Union[np.ndarray,
            np.ma.MaskedArray]]]: Aggregated results, as an array corresponding to the mode when
            ``with_uncertainty=False`` and as a dict otherwise including:
                - ``"loc"``: Aggregated mode of shape ``(n_samples, ...)``.
                - ``"uncertainty"``: Uncertainty values of shape ``(n_samples, ...)`` `.

        Raises:
            ValueError: If `y` dimensions are invalid or if `weights` length does not match `y`.
        """
        if not isinstance(y, list) or not all(
            isinstance(arr, (np.ndarray, np.ma.MaskedArray)) for arr in y
        ):
            raise TypeError("Input `y` must be a list of numpy.ndarray or numpy.ma.MaskedArray.")

        self._np = np
        is_masked = False
        if all(isinstance(pred, np.ma.MaskedArray) for pred in y):
            self._np = np.ma
            is_masked = True

        # Categorical probabilities (n_predictors, n_samples, ..., n_classes)
        y_proba_models = self._np.stack(y, axis=0)
        n_predictors = y_proba_models.shape[0]
        num_classes = y_proba_models.shape[-1]

        # Mode of the ensemble (n_samples, ...)
        y_mode_models = self._np.argmax(y_proba_models, axis=-1)

        weighted_counts = self._np.zeros_like(y_proba_models, dtype=np.float64).sum(axis=0)
        eye_arr = np.eye(num_classes, dtype=np.float64)
        for i in range(n_predictors):
            if weights is None:
                weighted_counts += eye_arr[y_mode_models[i]] / n_predictors
            else:
                weighted_counts += eye_arr[y_mode_models[i]] * weights[i]

        y_mode_ensemble = weighted_counts.argmax(axis=-1)
        if is_masked:
            mask = weighted_counts.sum(axis=-1).mask
            y_mode_ensemble = self._np.array(y_mode_ensemble, mask=mask)

        if not self.with_uncertainty:
            return y_mode_ensemble
        else:
            # Uncertainty of ensemble
            uncertainty = 1 - self._np.max(weighted_counts, axis=-1)

            return {
                "loc": y_mode_ensemble,
                "uncertainty": uncertainty,
            }

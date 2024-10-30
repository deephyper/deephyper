from typing import List

import numpy as np

from deephyper.ensemble.aggregator._aggregator import Aggregator


class ModeAggregator(Aggregator):
    """Aggregate the predictions using the mode of categorical distribution of each predictor in the ensemble.

    .. list-table::
        :widths: 25 25
        :header-rows: 1

        * - Array (Fixed Set)
          - MaskedArray
        * - ✅
          - ❌

    This aggregator is useful when the ensemble is composed of predictors that output categorical distributions. The mode of the ensemble is the mode of the modes of the predictors. This minimizes the 0-1 loss.
    """

    def aggregate(self, y: List, weights: List = None):
        """Aggregate the predictions using the mode of categorical distribution.

        Args:
            y (np.array): Predictions array of shape ``(n_predictors, n_samples, n_outputs)``.

            weights (list, optional): Weights of the predictors. Default is ``None``.

        Returns:
            np.array: Aggregated predictions of shape ``(n_samples, n_outputs)``.
        """

        # Categorical probabilities (n_predictors, n_samples, ..., n_classes)
        y_proba_models = np.asarray(y)
        n_predictors = y_proba_models.shape[0]
        n_samples = y_proba_models.shape[1]
        n_classes = y_proba_models.shape[-1]

        # Mode of the ensemble (n_samples, ...)
        y_mode_models = np.argmax(y_proba_models, axis=-1)

        counts = np.asarray(
            [
                np.bincount(y_mode_models[:, i], minlength=n_classes)
                for i in range(n_samples)
            ]
        )

        if weights is not None:
            counts = np.asarray(weights) * counts

        y_mode_ensemble = np.argmax(counts, axis=1)

        # Uncertainty of ensemble
        uncertainty = 1 - np.max(counts, axis=1) / n_predictors

        agg = {
            "loc": y_mode_ensemble,
            "uncertainty": uncertainty,
        }

        return agg

from typing import List

import numpy as np
import scipy.stats as ss

from deephyper.ensemble.aggregator._aggregator import Aggregator
from deephyper.ensemble.aggregator._mean import average


class MixedCategoricalAggregator(Aggregator):
    """Aggregate a set of categorical distributions.

    .. list-table::
        :widths: 25 25
        :header-rows: 1

        * - Array (Fixed Set)
          - MaskedArray
        * - ✅
          - ❌


    Args:
        uncertainty_method (str, optional): Method to compute the uncertainty. Can be either ``"confidence"`` or ``"entropy"``. Default is ``"confidence"``.

            - ``"confidence"``: The uncertainty is computed as ``1 - max(probability)`` of the aggregated categorical distribution of ensemble.
            - ``"entropy"``: The uncertainty is computed as the ``entropy`` of of the aggregated categorical distribution of ensemble.

        decomposed_uncertainty (bool, optional): If ``True``, the uncertainty of the ensemble is decomposed into aleatoric and epistemic components. Default is ``False``.
    """

    def __init__(
        self,
        uncertainty_method="confidence",
        decomposed_uncertainty: bool = False,
    ):
        assert uncertainty_method in ["confidence", "entropy"]
        self.uncertainty_method = uncertainty_method
        self.decomposed_uncertainty = decomposed_uncertainty

    def aggregate(self, y: List, weights: List = None):
        """Aggregate the predictions using the mode of categorical distribution.

        Args:
            y (np.array): Predictions array of shape ``(n_predictors, n_samples, n_outputs)``.

            weights (list, optional): Weights of the predictors. Default is ``None``.

        Returns:
            np.array: Aggregated predictions of shape ``(n_samples, n_outputs)``.
        """
        # Categorical probabilities (n_predictors, n_samples, ..., n_classes)
        y_proba_models = y
        y_proba_ensemble = average(y_proba_models, weights=weights, axis=0)

        agg = {
            "loc": y_proba_ensemble,
        }

        # Confidence of the ensemble: max probability of the ensemble
        if self.uncertainty_method == "confidence":
            uncertainty = 1 - np.max(y_proba_ensemble, axis=-1)

            if not self.decomposed_uncertainty:
                agg["uncertainty"] = uncertainty

            else:
                # Uncertainty of the mode: 1 - confidence(n_predictors, n_samples, ...)
                uncertainty_aleatoric = np.average(
                    1 - np.max(y_proba_models, axis=-1), weights=weights, axis=0
                )

                # TODO: looking at the decomposition of Domingo et al. it is possible that we should take into consideration the coef c1 and c2 to compute the epistemic uncertainty
                uncertainty_epistemic = np.maximum(
                    0, uncertainty - uncertainty_aleatoric
                )

                agg["uncertainty_aleatoric"] = uncertainty_aleatoric
                agg["uncertainty_epistemic"] = uncertainty_epistemic

        # Entropy of the ensemble
        elif self.uncertainty_method == "entropy":
            uncertainty = ss.entropy(y_proba_ensemble, axis=-1)

            if not self.decomposed_uncertainty:
                agg["uncertainty"] = uncertainty

            else:
                # Expectation over predictors in the ensemble
                expected_entropy = np.average(
                    ss.entropy(y_proba_models, axis=-1), weights=weights, axis=0
                )
                agg["uncertainty_aleatoric"] = expected_entropy
                agg["uncertainty_epistemic"] = np.maximum(
                    0, uncertainty - expected_entropy
                )

        return agg

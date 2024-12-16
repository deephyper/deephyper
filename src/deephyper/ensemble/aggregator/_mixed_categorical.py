from typing import Dict, List, Optional, Union

import numpy as np

from deephyper.ensemble.aggregator._aggregator import Aggregator


class MixedCategoricalAggregator(Aggregator):
    """Aggregate a set of categorical distributions, supporting uncertainty estimation.

    .. list-table::
        :widths: 25 25
        :header-rows: 1

        * - Array (Fixed Set)
          - MaskedArray
        * - ✅
          - ✅

    Args:
        uncertainty_method (str, optional): Method to compute the uncertainty.
            Choices are ``"confidence"`` or ``"entropy"``. Default is ``"confidence"``.
            - ``"confidence"``: Uncertainty is computed as ``1 - max(probability)``.
            - ``"entropy"``: Uncertainty is computed as the entropy of the categorical distribution.

        decomposed_uncertainty (bool, optional):
            If ``True``, decomposes uncertainty into aleatoric and epistemic components. Default is
            ``False``.
    """

    VALID_UNCERTAINTY_METHODS = {"confidence", "entropy"}

    def __init__(
        self,
        uncertainty_method: str = "confidence",
        decomposed_uncertainty: bool = False,
    ):
        if uncertainty_method not in self.VALID_UNCERTAINTY_METHODS:
            raise ValueError(
                f"Invalid uncertainty_method '{uncertainty_method}'. "
                f"Valid options are {self.VALID_UNCERTAINTY_METHODS}."
            )
        self.uncertainty_method = uncertainty_method
        self.decomposed_uncertainty = decomposed_uncertainty

    def aggregate(
        self, y: List[np.ndarray], weights: Optional[List[float]] = None
    ) -> Dict[str, Union[np.ndarray, np.ma.MaskedArray]]:
        """Aggregate predictions using the mode of categorical distributions.

        Args:
            y (List[np.ndarray]): List of categorical probability arrays of shape
                ``(n_predictors, n_samples, ..., n_classes)``.
            weights (Optional[List[float]]): Optional weights for the predictors.
                Must match the number of predictors. Default is ``None``.

        Returns:
            Dict[str, Union[np.ndarray, float]]: Aggregated results, including:
                - ``"loc"``: Aggregated categorical probabilities of shape ``(n_samples, ...,
                 n_classes)``.
                - ``"uncertainty"``: (Optional) Total uncertainty.
                - ``"uncertainty_aleatoric"``: (Optional) Aleatoric uncertainty.
                - ``"uncertainty_epistemic"``: (Optional) Epistemic uncertainty.

        Raises:
            ValueError: If `y` dimensions are invalid or if `weights` length does not match `y`.
        """
        if not isinstance(y, list) or not all(isinstance(arr, np.ndarray) for arr in y):
            raise TypeError("Input `y` must be a list of numpy.ndarray.")

        if weights is not None and len(weights) != len(y):
            raise ValueError("The length of `weights` must match the number of predictors in `y`.")

        self._np = np
        if all(isinstance(pred, np.ma.MaskedArray) for pred in y):
            self._np = np.ma

        # Stack predictions and compute ensemble probabilities
        y_proba_models = self._np.stack(
            y, axis=0
        )  # Shape: (n_predictors, n_samples, ..., n_classes)
        y_proba_ensemble = self._np.average(y_proba_models, weights=weights, axis=0)

        agg = {"loc": y_proba_ensemble}

        # Compute uncertainty
        if self.uncertainty_method == "confidence":
            self._compute_confidence_uncertainty(agg, y_proba_models, y_proba_ensemble, weights)
        elif self.uncertainty_method == "entropy":
            self._compute_entropy_uncertainty(agg, y_proba_models, y_proba_ensemble, weights)

        return agg

    def _compute_confidence_uncertainty(
        self,
        agg: Dict,
        y_proba_models: np.ndarray,
        y_proba_ensemble: np.ndarray,
        weights: Optional[List[float]],
    ):
        """Compute confidence-based uncertainty."""
        uncertainty = 1 - self._np.max(y_proba_ensemble, axis=-1)

        if not self.decomposed_uncertainty:
            agg["uncertainty"] = uncertainty
        else:
            uncertainty_aleatoric = self._np.average(
                1 - self._np.max(y_proba_models, axis=-1), weights=weights, axis=0
            )
            uncertainty_epistemic = self._np.maximum(0, uncertainty - uncertainty_aleatoric)

            agg["uncertainty_aleatoric"] = uncertainty_aleatoric
            agg["uncertainty_epistemic"] = uncertainty_epistemic

    def _compute_entropy_uncertainty(
        self,
        agg: Dict,
        y_proba_models: np.ndarray,
        y_proba_ensemble: np.ndarray,
        weights: Optional[List[float]],
    ):
        """Compute entropy-based uncertainty."""
        uncertainty = self._entropy(y_proba_ensemble, axis=-1)

        if not self.decomposed_uncertainty:
            agg["uncertainty"] = uncertainty
        else:
            expected_entropy = self._np.average(
                self._entropy(y_proba_models, axis=-1), weights=weights, axis=0
            )
            uncertainty_epistemic = self._np.maximum(0, uncertainty - expected_entropy)

            agg["uncertainty_aleatoric"] = expected_entropy
            agg["uncertainty_epistemic"] = uncertainty_epistemic

    def _entropy(self, prob, axis=None):
        eps = np.finfo(prob.dtype).eps
        return -self._np.sum(prob * self._np.log(prob + eps), axis=-1)

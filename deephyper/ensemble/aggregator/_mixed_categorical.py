import numpy as np
import scipy.stats as ss

from deephyper.ensemble.aggregator._aggregator import Aggregator


class MixedCategoricalAggregator(Aggregator):
    """Aggregate a set of categorical distributions.

    The return ``"uncertainty"`` is ``1 - confidence`` where ``confidence`` is the maximum probability of the ensemble.

    Args:
        decomposed_uncertainty (bool, optional): If ``True``, the uncertainty of the ensemble is decomposed into aleatoric and epistemic components. Default is ``False``.
    """

    def __init__(
        self, uncertainty_method="confidence", decomposed_uncertainty: bool = False
    ):
        assert uncertainty_method in ["confidence", "entropy"]
        self.uncertainty_method = uncertainty_method
        self.decomposed_uncertainty = decomposed_uncertainty

    def aggregate(self, y):
        """Aggregate the predictions using the mode of categorical distribution.

        Args:
            y (np.array): Predictions array of shape ``(n_predictors, n_samples, n_outputs)``.

        Returns:
            np.array: Aggregated predictions of shape ``(n_samples, n_outputs)``.
        """
        # Categorical probabilities (n_predictors, n_samples, ..., n_classes)
        y_proba_models = np.asarray(y)
        y_proba_ensemble = np.mean(y_proba_models, axis=0)
        n_classes = y_proba_models.shape[-1]  # For normalization with number of classes
        # n_classes = 1

        agg = {
            "loc": y_proba_ensemble,
        }

        # Confidence of the ensemble: max probability of the ensemble
        if self.uncertainty_method == "confidence":
            uncertainty = (1 - np.max(y_proba_ensemble, axis=-1)) * n_classes
            if not self.decomposed_uncertainty:
                agg["uncertainty"] = uncertainty
            else:
                # Uncertainty of the mode: 1 - confidence(n_predictors, n_samples, ...)
                y_aleatoric = (
                    np.mean(1 - np.max(y_proba_models, axis=-1), axis=0) * n_classes
                )
                y_epistemic = uncertainty - y_aleatoric

                agg["uncertainty_aleatoric"] = y_aleatoric
                agg["uncertainty_epistemic"] = y_epistemic

        # Entropy of the ensemble
        elif self.uncertainty_method == "entropy":
            uncertainty = ss.entropy(y_proba_ensemble, axis=-1)
            if not self.decomposed_uncertainty:
                agg["uncertainty"] = uncertainty
            else:
                # Expectation over predictors in the ensemble
                expected_entropy = np.mean(ss.entropy(y_proba_models, axis=-1), axis=0)
                agg["uncertainty_aleatoric"] = expected_entropy
                agg["uncertainty_epistemic"] = uncertainty - expected_entropy

        return agg

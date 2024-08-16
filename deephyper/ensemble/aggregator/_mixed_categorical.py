import numpy as np

from deephyper.ensemble.aggregator._aggregator import Aggregator


class MixedCategoricalAggregator(Aggregator):
    """Aggregate a set of categorical distributions.

    The return ``"uncertainty"`` is ``1 - confidence`` where ``confidence`` is the maximum probability of the ensemble.

    Args:
        decomposed_uncertainty (bool, optional): If ``True``, the uncertainty of the ensemble is decomposed into aleatoric and epistemic components. Default is ``False``.
    """

    def __init__(self, decomposed_uncertainty: bool = False):
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
        # n_classes = y_proba_models.shape[-1] # For normalization with number of classes
        n_classes = 1

        agg = {
            "loc": y_proba_ensemble,
        }

        if self.decomposed_uncertainty:
            # Mode of the ensemble (n_samples, ...)
            y_mode_ensemble = np.argmax(y_proba_ensemble, axis=-1)  # (n_samples,)

            # Uncertainty of the mode: 1 - confidence(n_predictors, n_samples, ...)
            y_aleatoric = (
                np.mean(1 - np.max(y_proba_models, axis=-1), axis=0) * n_classes
            )
            y_epistemic = (
                np.mean(
                    1
                    - np.take_along_axis(
                        y_proba_models,  # (n_predictors, n_samples, n_classes)
                        y_mode_ensemble.reshape(1, y_mode_ensemble.shape[0], 1),
                        axis=-1,
                    ),
                    axis=0,
                )
                * n_classes
            )
            agg["uncertainty_aleatoric"] = y_aleatoric
            agg["uncertainty_epistemic"] = y_epistemic
        else:
            agg["uncertainty"] = (1 - np.max(y_proba_ensemble, axis=-1)) * n_classes

        return agg

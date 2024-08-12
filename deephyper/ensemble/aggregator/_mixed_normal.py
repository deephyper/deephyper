import numpy as np

from deephyper.ensemble.aggregator._aggregator import Aggregator


class MixedNormalAggregator(Aggregator):
    """Aggregate a collection of predictions each representing a normal distribution.

    Args:
        weights (optional): a sequence of predictors' weight in the average. Defaults to ``None``.
    """

    def __init__(self, weights=None, decomposed_std=False):
        self.weights = weights
        self.decomposed_std = decomposed_std

    def aggregate(self, y):
        """Aggregate the predictions using the mean.

        Args:
            y (Dict): Predictions dictionary with keys ``loc`` for the mean and ``scale`` for the standard-deviation of each normal distribution, each with shape ``(n_models, n_samples, ..., n_outputs)``.

        Returns:
            Dict: Aggregated predictions with keys ``loc`` for the mean and ``scale`` for the standard-deviation of the mixture distribution, each with shape ``(n_samples, ..., n_outputs)``.
        """
        y_dict = {k: list() for k in y[0].keys()}
        for yi in y:
            for k, v in yi.items():
                y_dict[k].append(v)
        y = {k: np.asarray(v) for k, v in y_dict.items()}

        loc = y["loc"]
        scale = y["scale"]

        mean_loc = np.average(loc, axis=0, weights=self.weights)

        if self.decomposed_std:
            scale_aleatoric = np.sqrt(np.mean(np.square(scale), axis=0))
            scale_epistemic = np.sqrt(np.square(np.std(loc, axis=0)))
            agg = {
                "loc": mean_loc,
                "scale_aleatoric": scale_aleatoric,
                "scale_epistemic": scale_epistemic,
            }
        else:
            sum_loc_scale = np.square(loc) + np.square(scale)
            mean_scale = np.sqrt(
                np.average(sum_loc_scale, axis=0, weights=self.weights)
                - np.square(mean_loc)
            )
            agg = {"loc": mean_loc, "scale": mean_scale}

        return agg

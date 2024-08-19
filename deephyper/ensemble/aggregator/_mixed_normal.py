from typing import List

import numpy as np

from deephyper.ensemble.aggregator._aggregator import Aggregator


class MixedNormalAggregator(Aggregator):
    """Aggregate a collection of predictions each representing a normal distribution.

    Args:
        decomposed_scale (bool, optional): If ``True``, the scale of the mixture distribution is decomposed into aleatoric and epistemic components. Default is ``False``.
    """

    def __init__(self, decomposed_scale: bool = False):
        self.decomposed_scale = decomposed_scale

    def aggregate(self, y: List, weights: List = None):
        """Aggregate the predictions using the mean.

        Args:
            y (Dict): Predictions dictionary with keys ``loc`` for the mean and ``scale`` for the standard-deviation of each normal distribution, each with shape ``(n_predictors, n_samples, ..., n_outputs)``.

            weights (list, optional): Weights of the predictors. Default is ``None``.

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

        mean_loc = np.average(loc, weights=weights, axis=0)
        agg = {
            "loc": mean_loc,
        }

        if not self.decomposed_scale:
            sum_loc_scale = np.square(loc) + np.square(scale)
            mean_scale = np.sqrt(
                np.average(sum_loc_scale, weights=weights, axis=0) - np.square(mean_loc)
            )
            agg["scale"] = mean_scale

        else:
            # Here we assume that the mixture distribution is a normal distribution with a scale that is the sum of the aleatoric and epistemic scales. This is a significant approximation that could be improved by returning the true GMM.
            scale_aleatoric = np.sqrt(
                np.average(np.square(scale), weights=weights, axis=0)
            )
            scale_epistemic = np.sqrt(np.square(np.std(loc, axis=0)))
            agg["scale_aleatoric"] = scale_aleatoric
            agg["scale_epistemic"] = scale_epistemic

        return agg

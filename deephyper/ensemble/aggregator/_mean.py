import numpy as np

from deephyper.ensemble.aggregator._aggregator import Aggregator


class MeanAggregator(Aggregator):
    """Aggregate the predictions using the average.

    Args:
        weights (optional): a sequence of predictors' weight in the average. Defaults to ``None``.
    """

    def __init__(self, weights=None):
        self.weights = weights

    def aggregate(self, y):
        """Aggregate the predictions using the mean.

        Args:
            y (np.array): Predictions array of shape ``(n_models, n_samples, n_outputs)``.

        Returns:
            np.array: Aggregated predictions of shape ``(n_samples, n_outputs)``.
        """
        y = np.average(y, axis=0, weights=self.weights)
        return y

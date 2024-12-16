import abc

from typing import List


class Aggregator(abc.ABC):
    """Base class that represents an aggregation function of a set of predictors."""

    @abc.abstractmethod
    def aggregate(self, y: List, weights: List = None):
        """Aggregate the predictions from different predictors.

        Args:
            y (List): List of predictions from different models. It should be of shape
                ``(n_predictors, ...)``.

            weights (list, optional): Weights of the predictors. Default is ``None``.

        Returns:
            Any: Aggregated predictions.
        """

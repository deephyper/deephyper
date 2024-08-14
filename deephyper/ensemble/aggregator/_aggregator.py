import abc

from typing import List


class Aggregator(abc.ABC):
    """Base class that represents an aggregation function for the predictions of a set of predictors."""

    @abc.abstractmethod
    def aggregate(self, y: List):
        """Aggregate the predictions from different predictors.

        Args:
            y (List): List of predictions from different models. It should be of shape ``(n_predictors, ...)``.

        Returns:
            Any: Aggregated predictions.
        """

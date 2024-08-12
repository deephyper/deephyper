import numpy as np

from deephyper.ensemble.aggregator._aggregator import Aggregator


class ModeAggregator(Aggregator):
    """Aggregate the predictions using the mode of categorical distribution."""

    def aggregate(self, y):
        """Aggregate the predictions using the mode of categorical distribution.

        Args:
            y (np.array): Predictions array of shape ``(n_models, n_samples, n_outputs)``.

        Returns:
            np.array: Aggregated predictions of shape ``(n_samples, n_outputs)``.
        """
        # TODO: ...
        raise NotImplementedError

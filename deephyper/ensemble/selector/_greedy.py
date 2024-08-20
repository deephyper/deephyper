from typing import Callable, Sequence

import numpy as np

from deephyper.ensemble.aggregator._aggregator import Aggregator
from deephyper.ensemble.selector._selector import Selector


class GreedySelector(Selector):
    """Selection method implementing Greedy (a.k.a., Caruana) selection. This method iteratively and greedily selects the predictors that minimize the loss when aggregated together.

    Args:
        loss_func (Callable or Loss): a loss function that takes two arguments: the true target values and the predicted target values.

        aggregator (Aggregator): The aggregator to use to combine the predictions of the selected predictors.

        k (int, optional): The number of predictors to select. Defaults to ``5``.

        k_init (int, optional): Regularization parameter for greedy selection. It is the number of predictors to select in the initialization step. Defaults to ``1``.
    """

    def __init__(
        self, loss_func: Callable, aggregator: Aggregator, k: int = 5, k_init: int = 1
    ):
        super().__init__(loss_func)
        self.aggregator = aggregator
        self.k = k
        self.k_init = k_init

    def _aggregate(self, y_predictors):
        return self.aggregator.aggregate(y_predictors)

    def select(self, y, y_predictors) -> Sequence[int]:
        # Initialization
        losses = [self._evaluate(y, y_pred_i) for y_pred_i in y_predictors]
        selected_indices = np.argsort(losses)[: self.k_init].tolist()
        loss_min = self._evaluate(
            y, self._aggregate([y_predictors[i] for i in selected_indices])
        )
        n_predictors = len(y_predictors)

        # Greedy steps
        while len(np.unique(selected_indices)) < self.k:
            losses = [
                self._evaluate(
                    y,
                    self._aggregate(
                        [y_predictors[i] for i in selected_indices + [i]],
                    ),
                )
                for i in range(n_predictors)  # iterate over all models
            ]
            i_min_ = np.nanargmin(losses)
            loss_min_ = losses[i_min_]

            if loss_min_ < loss_min:
                if (
                    len(np.unique(selected_indices)) == 1
                    and selected_indices[0] == i_min_
                ):  # numerical errors...
                    break
                loss_min = loss_min_
                selected_indices.append(i_min_)
            else:
                break

        selected_indices, selected_indices_weights = np.unique(
            selected_indices, return_counts=True
        )
        selected_indices_weights = selected_indices_weights / np.sum(
            selected_indices_weights
        )
        return selected_indices.tolist(), selected_indices_weights.tolist()

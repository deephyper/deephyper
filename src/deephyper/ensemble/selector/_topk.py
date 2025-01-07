from typing import Callable, Sequence, Tuple

import numpy as np

from deephyper.ensemble.selector._selector import Selector
from deephyper.ensemble.loss import Loss


class TopKSelector(Selector):
    """Selection method implementing Top-K selection.

    This method selects the K predictors with the lowest loss.

    Args:
        loss_func (Callable or Loss): a loss function that takes two arguments: the true target
            values and the predicted target values.
        k (int, optional): The number of predictors to select. Defaults to ``5``.
    """

    def __init__(self, loss_func: Callable | Loss, k: int = 5):
        super().__init__(loss_func)
        self.k = k

    def select(self, y, y_predictors) -> Tuple[Sequence[int], Sequence[float]]:
        losses = [self._evaluate(y, y_pred_i) for y_pred_i in y_predictors]
        selected_indices = np.argsort(losses, axis=0)[: self.k].reshape(-1).tolist()
        selected_indices_weights = [1.0] * len(selected_indices)
        return selected_indices, selected_indices_weights

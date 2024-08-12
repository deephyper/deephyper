import abc
from typing import Callable, Sequence

import numpy as np

from deephyper.predictor import Predictor
from deephyper.ensemble.aggregator._aggregator import Aggregator


class Selector(abc.ABC):

    def __init__(self, loss_func: Callable):
        self.loss_func = loss_func

    @abc.abstractmethod
    def select(self, y, y_predictors) -> Sequence[int]:
        pass


class TopKSelector(Selector):
    def __init__(self, loss_func: Callable, k: int = 5):
        super().__init__(loss_func)
        self.k = k

    def select(self, y, y_predictors) -> Sequence[int]:
        losses = [np.mean(self.loss_func(y, y_pred_i)) for y_pred_i in y_predictors]
        selected_indices = np.argsort(losses, axis=0)[: self.k].reshape(-1).tolist()
        return selected_indices


class GreedySelector(Selector):
    def __init__(self, loss_func: Callable, aggregator: Aggregator, k: int = 5):
        super().__init__(loss_func)
        self.aggregator = aggregator
        self.k = k

    def select(self, y, y_predictors) -> Sequence[int]:
        # Initialization
        losses = [np.mean(self.loss_func(y, y_pred_i)) for y_pred_i in y_predictors]
        i_min = np.nanargmin(losses)
        loss_min = losses[i_min]
        selected_indices = [i_min]
        n_predictors = len(y_predictors)

        # Greedy steps
        while len(np.unique(selected_indices)) < self.k:
            losses = [
                self.loss_func(
                    y,
                    self.aggregator.aggregate(
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
                    return selected_indices
                loss_min = loss_min_
                selected_indices.append(i_min_)
            else:
                break

        return selected_indices

from typing import Callable, List, Sequence

import numpy as np
from sklearn.utils import check_random_state

from deephyper.ensemble.aggregator._aggregator import Aggregator
from deephyper.ensemble.selector._selector import Selector


class GreedySelector(Selector):
    """Selection method implementing Greedy (a.k.a., Caruana) selection.

    This method iteratively and greedily selects the predictors that minimize
    the loss when aggregated together.

    Args:
        loss_func (Callable or Loss): a loss function that takes two arguments: the true target
            values and the predicted target values.
        aggregator (Aggregator): The aggregator to use to combine the predictions of the selected
            predictors.
        k (int, optional): The number of unique predictors to select for the ensemble. Defaults to
            ``5``.
        k_init (int, optional): Regularization parameter for greedy selection. It is the number of
            predictors to select in the initialization step. Defaults to ``1``.
        max_it (int, optional): Maximum number of iterations which also corresponds to the number
            of non-unique predictors added to the ensemble. Defaults to ``-1``.
        eps_tol (float, optional): Tolerance for the stopping criterion. Defaults to ``1e-3``.
        with_replacement (bool, optional): Performs greedy selection with replacement of models
            already selected. Defaults to ``True``.
        early_stopping (bool, optional): Stops the ensemble selection as soon as the loss stops
            improving. Defaults to ``True``.
        bagging (bool, optional): Performs boostrap resampling of available predictors at each
            iteration. This can be particularly useful when the dataset used for selection is
            small. Defaults to ``False``.
        verbose (bool, optional):
            Turns on the verbose mode. Defaults to ``False``.
    """

    def __init__(
        self,
        loss_func: Callable,
        aggregator: Aggregator,
        k: int = 5,
        k_init: int = 5,
        max_it: int = -1,
        eps_tol: float = 1e-3,
        with_replacement: bool = True,
        early_stopping: bool = True,
        bagging: bool = False,
        random_state=None,
        verbose: bool = False,
    ):
        super().__init__(loss_func)
        self.aggregator = aggregator
        self.k = k
        self.k_init = k_init
        self.max_it = max_it
        self.eps_tol = eps_tol
        self.with_replacement = with_replacement
        self.early_stopping = early_stopping
        self.bagging = bagging
        self.random_state = check_random_state(random_state)
        self.verbose = verbose

    def _aggregate(self, y_predictors: np.ndarray, weights: List = None):
        return self.aggregator.aggregate(y_predictors, weights)

    def select(self, y, y_predictors) -> Sequence[int]:
        # Initialization
        losses = [self._evaluate(y, y_pred_i) for y_pred_i in y_predictors]
        selected_indices = np.argsort(losses)[: self.k_init].tolist()
        selected_indices_weights = [1 / self.k_init] * self.k_init
        loss_min = self._evaluate(y, self._aggregate([y_predictors[i] for i in selected_indices]))
        n_predictors = len(y_predictors)
        bagged_predictors = None

        if self.verbose:
            tmp = [losses[i] for i in selected_indices]
            print(f"Ensemble initialized with {selected_indices} with loss {tmp}")

        # Greedy steps
        it = 0
        while (self.max_it < 0 or it < self.max_it) and len(np.unique(selected_indices)) < self.k:
            losses = []

            if self.bagging:
                bagged_predictors = np.unique(
                    self.random_state.randint(low=0, high=n_predictors, size=n_predictors)
                )

            for i in range(n_predictors):
                # Applying conditions that ignore some indices in the selection
                if len(selected_indices) == 1 and i in selected_indices:
                    losses.append(np.nan)
                    continue

                if not self.with_replacement and i in selected_indices:
                    losses.append(np.nan)
                    continue

                if self.bagging and i not in bagged_predictors:
                    losses.append(np.nan)
                    continue

                indices_ = selected_indices + [i]
                indices_, indices_weights_ = np.unique(indices_, return_counts=True)
                indices_weights_ = indices_weights_ / np.sum(indices_weights_)
                y_ = [y_predictors[i] for i in indices_]
                score = self._evaluate(
                    y,
                    self._aggregate(y_, indices_weights_),
                )
                losses.append(score)

            i_min_ = np.nanargmin(losses)
            loss_min_ = losses[i_min_]
            it += 1

            # The second condition is related to numerical errors
            if (self.early_stopping and loss_min_ >= (loss_min - self.eps_tol)) or (
                len(np.unique(selected_indices)) == 1 and selected_indices[0] == i_min_
            ):
                if self.verbose:
                    print(f"Step {it}, ensemble selection stopped")
                break

            loss_min = loss_min_
            selected_indices.append(i_min_)

            if self.verbose:
                print(
                    f"Step {it}, ensemble is {selected_indices}, new member {i_min_} with"
                    f" loss {loss_min}"
                )

        selected_indices, selected_indices_weights = np.unique(selected_indices, return_counts=True)
        selected_indices_weights = selected_indices_weights / np.sum(selected_indices_weights)

        if self.verbose:
            print(
                f"After {it} steps, the final ensemble is {selected_indices} with "
                f"weights {selected_indices_weights}"
            )

        return selected_indices.tolist(), selected_indices_weights.tolist()

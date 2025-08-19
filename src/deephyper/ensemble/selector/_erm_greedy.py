from typing import Callable, List

import numpy as np
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import QuantileTransformer
from sklearn.utils import check_random_state
from sklearn.compose import TransformedTargetRegressor

from deephyper.ensemble.aggregator._aggregator import Aggregator
from deephyper.ensemble.selector._selector import Selector


class ERMGreedySelector(Selector):
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

        self.erm_model = ExtraTreesRegressor(min_samples_leaf=4)

        # self.erm_model = LinearRegression()
        # self.erm_model = Pipeline(
        #     [
        #         # ("poly", PolynomialFeatures(degree=2)),
        #         (
        #             "linear",
        #             # TransformedTargetRegressor(Ridge(), func=np.log1p, inverse_func=np.expm1),
        #             TransformedTargetRegressor(
        #                 RandomForestRegressor(),
        #                 # Ridge(),
        #                 # transformer=QuantileTransformer(
        #                 #     n_quantiles=1000, output_distribution="uniform"
        #                 # ),
        #             ),
        #         ),
        #     ]
        # )

    def _aggregate(self, y_predictors: np.ndarray, weights: List = None):
        return self.aggregator.aggregate(y_predictors, weights)

    def fit(self, X, y, y_predictors):
        # Creates the Error model
        # Inputs: (X, y)
        # Output: Loss(y)
        if isinstance(y_predictors[0], dict):
            X_erm = np.concatenate(
                [
                    np.concatenate((X, y_pred_i["loc"], y_pred_i["scale"]), axis=1)
                    for y_pred_i in y_predictors
                ],
                axis=0,
            )
        else:
            X_erm = np.concatenate(
                [np.concatenate((X, y_pred_i), axis=1) for y_pred_i in y_predictors],
                axis=0,
            )
        y_erm = np.concatenate(
            [self.loss_func(y, y_pred_i) for y_pred_i in y_predictors],
            axis=0,
        ).reshape(-1)
        print("min:", np.min(y_erm))
        print("max:", np.max(y_erm))
        # print(f"{np.shape(X_erm)=}")
        # print(f"{np.shape(y_erm)=}")
        X_train, X_valid, y_train, y_valid = train_test_split(X_erm, y_erm, test_size=0.33)
        q1 = np.quantile(y_erm, q=0.25)
        q2 = np.quantile(y_erm, q=0.5)
        q3 = np.quantile(y_erm, q=0.75)
        print(f"{q1:.2f} | {q2:.2f} | {q3:.3f}")
        threshold = q3 + 1.5 * (q3 - q1)
        y_train[y_train > threshold] = threshold
        y_valid[y_valid > threshold] = threshold
        self.erm_model.fit(X_train, y_train)
        r2_score_train = self.erm_model.score(X_train, y_train)
        r2_score_valid = self.erm_model.score(X_valid, y_valid)
        print(f"{r2_score_train=:.3f}")
        print(f"{r2_score_valid=:.3f}")

        # import matplotlib.pyplot as plts

        # plt.close("all")
        # plt.figure()
        # plt.boxplot(y_erm)
        # plt.show()

        # y_valid_pred = self.erm_model.predict(X_valid)
        # plt.figure()
        # plt.scatter(y_valid, y_valid_pred)
        # plt.xlabel("True")
        # plt.ylabel("Pred")
        # plt.show()

    def _evaluate(self, X, y_pred) -> float:
        if isinstance(y_pred, dict):
            if "scale_aleatoric" in y_pred and "scale_epistemic" in y_pred:
                scale = np.sqrt(y_pred["scale_aleatoric"] ** 2 + y_pred["scale_epistemic"] ** 2)
                X_erm = np.concatenate((X, y_pred["loc"], scale), axis=1)
            else:
                X_erm = np.concatenate((X, y_pred["loc"], y_pred["scale"]), axis=1)
        else:
            X_erm = np.concatenate((X, y_pred), axis=1)
        return self._reduce(self.erm_model.predict(X_erm))

    def select(self, X, y_predictors) -> tuple[list[int], list[float]]:
        # Initialization
        losses = [self._evaluate(X, y_pred_i) for y_pred_i in y_predictors]
        selected_indices = np.argsort(losses)[: self.k_init].tolist()
        selected_indices_weights = [1 / self.k_init] * self.k_init
        loss_min = self._evaluate(X, self._aggregate([y_predictors[i] for i in selected_indices]))
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
                    X,
                    self._aggregate(y_, indices_weights_),
                )
                losses.append(score)

            i_min_ = int(np.nanargmin(losses))
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

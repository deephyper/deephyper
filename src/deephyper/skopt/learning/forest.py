import threading

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor as _sk_ExtraTreesRegressor
from sklearn.ensemble._forest import DecisionTreeRegressor, ForestRegressor

from deephyper.skopt.joblib import Parallel, delayed


def _accumulate_prediction_disentangled_v1(tree, X, min_variance, out, lock):
    """This is a utility function for joblib's Parallel.

    It can't go locally in ForestClassifier or ForestRegressor, because joblib
    complains that it cannot pickle it when placed there.
    """
    mean_tree = tree.predict(X).T
    var_tree = tree.tree_.impurity[tree.apply(X)]

    # This rounding off is done in accordance with the
    # adjustment done in section 4.3.3
    # of http://arxiv.org/pdf/1211.0906v2.pdf to account
    # for cases such as leaves with 1 sample in which there
    # is zero variance.
    var_tree = np.maximum(var_tree, min_variance)

    with lock:
        out[0] += mean_tree
        out[1] += var_tree
        out[2] += mean_tree**2


def _return_mean_and_std_distentangled_v1(X, n_outputs, trees, min_variance, n_jobs):
    """Returns `std(Y | X)`.

    Can be calculated by E[Var(Y | Tree)] + Var(E[Y | Tree]) where
    P(Tree) is `1 / len(trees)`.

    Parameters
    ----------
    X : array-like, shape=(n_samples, n_features)
        Input data.

    n_outputs: int.
        Number of outputs.

    trees : list, shape=(n_estimators,)
        List of fit sklearn trees as obtained from the ``estimators_``
        attribute of a fit RandomForestRegressor or ExtraTreesRegressor.

    predictions : array-like, shape=(n_samples,)
        Prediction of each data point as returned by RandomForestRegressor
        or ExtraTreesRegressor.

    Returns:
    -------
    std : array-like, shape=(n_samples,)
        Standard deviation of `y` at `X`. If criterion
        is set to "mse", then `std[i] ~= std(y | X[i])`.

    """
    # This derives std(y | x) as described in 4.3.2 of arXiv:1211.0906

    mean = np.zeros((n_outputs, len(X)))
    std_al = np.zeros((n_outputs, len(X)))
    std_ep = np.zeros((n_outputs, len(X)))

    # Parallel loop
    lock = threading.Lock()
    Parallel(n_jobs=n_jobs, verbose=0, require="sharedmem")(
        delayed(_accumulate_prediction_disentangled_v1)(
            tree, X, min_variance, [mean, std_al, std_ep], lock
        )
        for tree in trees
    )

    mean, std_al, std_ep = mean.T, std_al.T, std_ep.T
    mean /= len(trees)

    std_al /= len(trees)
    std_ep = std_ep / len(trees) - mean**2

    std_al[std_al <= 0.0] = 0.0
    std_al **= 0.5

    std_ep[std_ep <= 0.0] = 0.0
    std_ep **= 0.5

    return mean.reshape(-1), std_al.reshape(-1), std_ep.reshape(-1)

def _accumulate_prediction_disentangled_v2(tree, X, min_variance, out):
    """This is a utility function for joblib's Parallel.

    It can't go locally in ForestClassifier or ForestRegressor, because joblib
    complains that it cannot pickle it when placed there.
    """
    # Compute leaf indices once
    leaf_idx = tree.apply(X)

    # Mean prediction from leaves
    mean_tree = tree.tree_.value[leaf_idx].ravel()

    # Impurity (e.g. variance for regression)
    var_tree = tree.tree_.impurity[leaf_idx].ravel()

    # This rounding off is done in accordance with the
    # adjustment done in section 4.3.3
    # of http://arxiv.org/pdf/1211.0906v2.pdf to account
    # for cases such as leaves with 1 sample in which there
    # is zero variance.
    var_tree[var_tree < min_variance] = min_variance

    out[0] += mean_tree
    out[1] += var_tree
    out[2] += mean_tree**2

def _return_mean_and_std_distentangled_v2(X, n_outputs, trees, min_variance, n_jobs):
    """Returns `std(Y | X)`.

    Can be calculated by E[Var(Y | Tree)] + Var(E[Y | Tree]) where
    P(Tree) is `1 / len(trees)`.

    Parameters
    ----------
    X : array-like, shape=(n_samples, n_features)
        Input data.

    n_outputs: int.
        Number of outputs.

    trees : list, shape=(n_estimators,)
        List of fit sklearn trees as obtained from the ``estimators_``
        attribute of a fit RandomForestRegressor or ExtraTreesRegressor.

    predictions : array-like, shape=(n_samples,)
        Prediction of each data point as returned by RandomForestRegressor
        or ExtraTreesRegressor.

    Returns:
    -------
    std : array-like, shape=(n_samples,)
        Standard deviation of `y` at `X`. If criterion
        is set to "mse", then `std[i] ~= std(y | X[i])`.

    """
    # This derives std(y | x) as described in 4.3.2 of arXiv:1211.0906

    n = len(trees)
    mean = np.zeros((len(X),))
    std_al = np.zeros((len(X),))
    std_ep = np.zeros((len(X),))

    # Parallel loop
    for tree in trees:
        _accumulate_prediction_disentangled_v2(tree, X, min_variance, [mean, std_al, std_ep])

    mean /= n

    std_al /= n
    std_ep = std_ep / n - mean**2

    std_al[std_al <= 0.0] = 0.0
    std_al **= 0.5

    std_ep[std_ep <= 0.0] = 0.0
    std_ep **= 0.5

    return mean, std_al, std_ep

def _return_mean_and_std_distentangled(X, n_outputs, trees, min_variance, n_jobs):
    if n_jobs == 1:
        return _return_mean_and_std_distentangled_v2(X, n_outputs, trees, min_variance, n_jobs)
    else:
        return _return_mean_and_std_distentangled_v1(X, n_outputs, trees, min_variance, n_jobs)
    
def _accumulate_prediction(tree, X, min_variance, out, lock):
    """This is a utility function for joblib's Parallel.

    It can't go locally in ForestClassifier or ForestRegressor, because joblib
    complains that it cannot pickle it when placed there.
    """
    mean_tree = tree.predict(X).T
    var_tree = tree.tree_.impurity[tree.apply(X)]

    # This rounding off is done in accordance with the
    # adjustment done in section 4.3.3
    # of http://arxiv.org/pdf/1211.0906v2.pdf to account
    # for cases such as leaves with 1 sample in which there
    # is zero variance.
    var_tree = np.maximum(var_tree, min_variance) + np.square(mean_tree)

    with lock:
        out[0] += mean_tree
        out[1] += var_tree


def _return_mean_and_std(X, n_outputs, trees, min_variance, n_jobs):
    """Returns `std(Y | X)`.

    Can be calculated by E[Var(Y | Tree)] + Var(E[Y | Tree]) where
    P(Tree) is `1 / len(trees)`.

    Parameters
    ----------
    X : array-like, shape=(n_samples, n_features)
        Input data.

    n_outputs: int.
        Number of outputs.

    trees : list, shape=(n_estimators,)
        List of fit sklearn trees as obtained from the ``estimators_``
        attribute of a fit RandomForestRegressor or ExtraTreesRegressor.

    predictions : array-like, shape=(n_samples,)
        Prediction of each data point as returned by RandomForestRegressor
        or ExtraTreesRegressor.

    Returns:
    -------
    std : array-like, shape=(n_samples,)
        Standard deviation of `y` at `X`. If criterion
        is set to "mse", then `std[i] ~= std(y | X[i])`.

    """
    # This derives std(y | x) as described in 4.3.2 of arXiv:1211.0906

    mean = np.zeros((n_outputs, len(X)))
    std = np.zeros((n_outputs, len(X)))

    # Parallel loop
    lock = threading.Lock()
    Parallel(n_jobs=n_jobs, verbose=0, require="sharedmem")(
        delayed(_accumulate_prediction)(tree, X, min_variance, [mean, std], lock)
        for tree in trees
    )

    mean, std = mean.T, std.T
    mean /= len(trees)
    std /= len(trees)
    std = np.sqrt(np.maximum(std - np.square(mean), 0.0))

    return mean.reshape(-1), std.reshape(-1)


class RandomForestRegressor(ForestRegressor):
    """RandomForestRegressor that supports conditional standard deviation computation.

    Args:
        n_estimators (int, optional): The number of trees in the forest.
            Defaults to ``100``.

        criterion (str, optional): The function to measure the quality of a split.
            Supported criteria are:
            - ``"mse"``: mean squared error (variance reduction)
            - ``"mae"``: mean absolute error  
            Defaults to ``"mse"``.

        max_features (int | float | str | None, optional): The number of features
            to consider when looking for the best split. Defaults to ``"1.0"``.

            - If int, consider ``max_features`` features at each split.
            - If float, treat as a percentage: ``int(max_features * n_features)``.
            - If ``"sqrt"``, use ``sqrt(n_features)``.
            - If ``"log2"``, use ``log2(n_features)``.
            - If ``None``, use all features.

            Note:
                The search for a split does not stop until at least one valid
                partition of the node samples is found, even if this requires
                inspecting more than ``max_features`` features.

        max_depth (int | None, optional): Maximum depth of the tree. If None, nodes
            expand until all leaves are pure or contain fewer than
            `min_samples_split` samples. Defaults to`` None``.

        min_samples_split (int | float, optional): Minimum number of samples
            required to split an internal node. Defaults to ``2``.

            - If int, use the exact number.
            - If float, interpret as a percentage:
            `ceil(min_samples_split * n_samples)`.

        min_samples_leaf (int | float, optional): Minimum number of samples
            required at a leaf node. Defaults to ``1``.

            - If int, use the exact number.
            - If float, interpret as a percentage: ``ceil(min_samples_leaf * n_samples)``.

        min_weight_fraction_leaf (float, optional): Minimum weighted fraction of
            the total sample weight required at a leaf node. Defaults to ``0.0``.

        max_leaf_nodes (int | None, optional): Grow trees with at most
            ``max_leaf_nodes`` in best-first fashion. If None, unlimited.
            Defaults to ``None``.

        min_impurity_decrease (float, optional): A node will be split if the
            impurity decrease is greater than or equal to this value. Defaults to 0.0.

            Weighted impurity decrease::

                N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

            where ``N`` is total weighted samples, ``N_t`` samples at current node,
            ``N_t_L`` left child, and ``N_t_R`` right child.

        bootstrap (bool, optional): Whether bootstrap samples are used when
            building trees. Defaults to ``True``.

        oob_score (bool, optional): Whether to use out-of-bag samples to estimate
            R² on unseen data. Defaults to ``False``.

        n_jobs (int, optional): Number of parallel jobs for ``fit`` and ``predict``.
            If -1, use all cores. Defaults to ``1``.

        random_state (int | RandomState | None, optional): Seed or random number
            generator. Defaults to ``None``.

        verbose (int, optional): Verbosity level of the tree-building process.
            Defaults to ``0``.

        warm_start (bool, optional): If ``True``, reuse solution from previous call to
            ``fit`` and add more estimators. Otherwise fit a new forest. Defaults to
            ``False``.

        splitter (str): The splitter strategy in ``["random", "best"]``. Defaults 
            to ``"best"``.

    Attributes:
        estimators_ (list[DecisionTreeRegressor]): Fitted sub-estimators.
        feature_importances_ (ndarray): Feature importances, shape (n_features,).
        n_features_ (int): Number of features at `fit` time.
        n_outputs_ (int): Number of outputs at `fit` time.
        oob_score_ (float): Out-of-bag R² score.
        oob_prediction_ (ndarray): Out-of-bag predictions, shape (n_samples,).

    Notes:
        The default hyperparameters (e.g., ``max_depth``, ``min_samples_leaf``)
        result in fully grown, unpruned trees, which may become large in memory.
        Consider adjusting these values to reduce complexity.

        Features are always randomly permuted at each split. Therefore, the best
        split may vary even with identical training data, ``max_features=n_features``,
        and ``bootstrap=False``. To ensure deterministic behavior, set
        ``random_state``.

    References:
        Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5-32.
    """

    def __init__(
        self,
        n_estimators=100,
        *,
        criterion="squared_error",
        max_depth=None,
        min_samples_split=10,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=1.0,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        ccp_alpha=0.0,
        max_samples=None,
        min_variance=0.0,
        splitter="best",
    ):
        super().__init__(
            # !keyword-argument changing from sklearn==1.2.0, positional fixed it!
            DecisionTreeRegressor(),
            n_estimators=n_estimators,
            estimator_params=(
                "criterion",
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "min_weight_fraction_leaf",
                "max_features",
                "max_leaf_nodes",
                "min_impurity_decrease",
                "random_state",
                "ccp_alpha",
                "splitter",
            ),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            max_samples=max_samples,
        )

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha

        self.min_variance = min_variance
        self.splitter = splitter

    def predict(self, X, return_std=False, disentangled_std=False):
        """Predict continuous output for X.

        Args:
        X : array of shape = (n_samples, n_features)
            Input data.

        return_std : boolean
            Whether or not to return the standard deviation.

        Returns:
        predictions : array-like of shape = (n_samples,)
            Predicted values for X. If criterion is set to "mse",
            then `predictions[i] ~= mean(y | X[i])`.

        std : array-like of shape=(n_samples,)
            Standard deviation of `y` at `X`. If criterion
            is set to "mse", then `std[i] ~= std(y | X[i])`.

        disentangled_std : the std is returned disentangled between aleatoric and epistemic.
        """
        if return_std:
            if self.criterion != "squared_error":
                raise ValueError(
                    "Expected impurity to be 'squared_error', got %s instead"
                    % self.criterion
                )
            if disentangled_std:
                mean, std_al, std_ep = _return_mean_and_std_distentangled(
                    X, self.n_outputs_, self.estimators_, self.min_variance, self.n_jobs
                )
                return mean, std_al, std_ep
            else:
                mean, std = _return_mean_and_std(
                    X, self.n_outputs_, self.estimators_, self.min_variance, self.n_jobs
                )
                return mean, std
        else:
            mean = super(RandomForestRegressor, self).predict(X)

            return mean


class ExtraTreesRegressor(_sk_ExtraTreesRegressor):
    """ExtraTreesRegressor that supports conditional standard deviation.

    Args:
        n_estimators (int, optional): The number of trees in the forest.
            Defaults to ``100``.

        criterion (str, optional): The function to measure the quality of a split.
            Supported criteria are:
            - ``"mse"``: mean squared error (variance reduction)
            - ``"mae"``: mean absolute error  
            Defaults to ``"mse"``.

        max_features (int | float | str | None, optional): The number of features
            to consider when looking for the best split. Defaults to ``"1.0"``.

            - If int, consider ``max_features`` features at each split.
            - If float, treat as a percentage: ``int(max_features * n_features)``.
            - If ``"sqrt"``, use ``sqrt(n_features)``.
            - If ``"log2"``, use ``log2(n_features)``.
            - If ``None``, use all features.

            Note:
                The search for a split does not stop until at least one valid
                partition of the node samples is found, even if this requires
                inspecting more than ``max_features`` features.

        max_depth (int | None, optional): Maximum depth of the tree. If None, nodes
            expand until all leaves are pure or contain fewer than
            `min_samples_split` samples. Defaults to`` None``.

        min_samples_split (int | float, optional): Minimum number of samples
            required to split an internal node. Defaults to ``2``.

            - If int, use the exact number.
            - If float, interpret as a percentage:
            `ceil(min_samples_split * n_samples)`.

        min_samples_leaf (int | float, optional): Minimum number of samples
            required at a leaf node. Defaults to ``1``.

            - If int, use the exact number.
            - If float, interpret as a percentage: ``ceil(min_samples_leaf * n_samples)``.

        min_weight_fraction_leaf (float, optional): Minimum weighted fraction of
            the total sample weight required at a leaf node. Defaults to ``0.0``.

        max_leaf_nodes (int | None, optional): Grow trees with at most
            ``max_leaf_nodes`` in best-first fashion. If None, unlimited.
            Defaults to ``None``.

        min_impurity_decrease (float, optional): A node will be split if the
            impurity decrease is greater than or equal to this value. Defaults to 0.0.

            Weighted impurity decrease::

                N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

            where ``N`` is total weighted samples, ``N_t`` samples at current node,
            ``N_t_L`` left child, and ``N_t_R`` right child.

        bootstrap (bool, optional): Whether bootstrap samples are used when
            building trees. Defaults to ``True``.

        oob_score (bool, optional): Whether to use out-of-bag samples to estimate
            R² on unseen data. Defaults to ``False``.

        n_jobs (int, optional): Number of parallel jobs for ``fit`` and ``predict``.
            If -1, use all cores. Defaults to ``1``.

        random_state (int | RandomState | None, optional): Seed or random number
            generator. Defaults to ``None``.

        verbose (int, optional): Verbosity level of the tree-building process.
            Defaults to ``0``.

        warm_start (bool, optional): If ``True``, reuse solution from previous call to
            ``fit`` and add more estimators. Otherwise fit a new forest. Defaults to
            ``False``.

    Attributes:
        estimators_ (list[DecisionTreeRegressor]): Fitted sub-estimators.
        feature_importances_ (ndarray): Feature importances, shape (n_features,).
        n_features_ (int): Number of features at `fit` time.
        n_outputs_ (int): Number of outputs at `fit` time.
        oob_score_ (float): Out-of-bag R² score.
        oob_prediction_ (ndarray): Out-of-bag predictions, shape (n_samples,).

    Notes:
        The default hyperparameters (e.g., ``max_depth``, ``min_samples_leaf``)
        result in fully grown, unpruned trees, which may become large in memory.
        Consider adjusting these values to reduce complexity.

        Features are always randomly permuted at each split. Therefore, the best
        split may vary even with identical training data, ``max_features=n_features``,
        and ``bootstrap=False``. To ensure deterministic behavior, set
        ``random_state``.

    References:
        Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5-32.
    """

    def __init__(
        self,
        n_estimators=100,
        criterion="squared_error",
        max_depth=None,
        min_samples_split=10,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=1.0,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=False,
        oob_score=False,
        n_jobs=1,
        random_state=None,
        verbose=0,
        warm_start=False,
        min_variance=0.0,
        max_samples=None,
    ):
        self.min_variance = min_variance
        super(ExtraTreesRegressor, self).__init__(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            max_samples=max_samples,
        )

    def predict(self, X, return_std=False, disentangled_std=False):
        """Predict continuous output for X.

        Args:
        X : array-like of shape=(n_samples, n_features)
            Input data.

        return_std : boolean
            Whether or not to return the standard deviation.

        Returns:
        predictions : array-like of shape=(n_samples,)
            Predicted values for X. If criterion is set to "squared_error",
            then `predictions[i] ~= mean(y | X[i])`.

        std : array-like of shape=(n_samples,)
            Standard deviation of `y` at `X`. If criterion
            is set to "squared_error", then `std[i] ~= std(y | X[i])`.
        """
        if return_std:
            if self.criterion != "squared_error":
                raise ValueError(
                    "Expected impurity to be 'squared_error', got %s instead"
                    % self.criterion
                )
            if disentangled_std:
                mean, std_al, std_ep = _return_mean_and_std_distentangled(
                    X, self.n_outputs_, self.estimators_, self.min_variance, self.n_jobs
                )
                return mean, std_al, std_ep
            else:
                mean, std = _return_mean_and_std(
                    X, self.n_outputs_, self.estimators_, self.min_variance, self.n_jobs
                )
                return mean, std
        else:
            mean = super(ExtraTreesRegressor, self).predict(X)

            return mean

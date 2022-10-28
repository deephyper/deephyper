"""
This module provides ``problem_autosklearn1`` and ``run_autosklearn`` for regression tasks.
"""
import warnings
from inspect import signature

import ConfigSpace as cs
from deephyper.problem import HpProblem
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor


def minmaxstdscaler() -> Pipeline:
    """MinMax preprocesssing followed by Standard normalization.

    Returns:
        Pipeline: a pipeline with two steps ``[MinMaxScaler, StandardScaler]``.
    """
    preprocessor = Pipeline(
        [
            ("minmaxscaler", MinMaxScaler()),
            ("stdscaler", StandardScaler()),
        ]
    )
    return preprocessor


REGRESSORS = {
    "RandomForest": RandomForestRegressor,
    "Linear": LinearRegression,
    "AdaBoost": AdaBoostRegressor,
    "KNeighbors": KNeighborsRegressor,
    "MLP": MLPRegressor,
    "SVR": SVR,
    "XGBoost": XGBRegressor,
}


problem_autosklearn1 = HpProblem()

regressor = problem_autosklearn1.add_hyperparameter(
    name="regressor",
    value=["RandomForest", "Linear", "AdaBoost", "KNeighbors", "MLP", "SVR", "XGBoost"],
)

# n_estimators
n_estimators = problem_autosklearn1.add_hyperparameter(
    name="n_estimators", value=(1, 2000, "log-uniform")
)

cond_n_estimators = cs.OrConjunction(
    cs.EqualsCondition(n_estimators, regressor, "RandomForest"),
    cs.EqualsCondition(n_estimators, regressor, "AdaBoost"),
)

problem_autosklearn1.add_condition(cond_n_estimators)

# max_depth
max_depth = problem_autosklearn1.add_hyperparameter(
    name="max_depth", value=(2, 100, "log-uniform")
)

cond_max_depth = cs.EqualsCondition(max_depth, regressor, "RandomForest")

problem_autosklearn1.add_condition(cond_max_depth)

# n_neighbors
n_neighbors = problem_autosklearn1.add_hyperparameter(
    name="n_neighbors", value=(1, 100)
)

cond_n_neighbors = cs.EqualsCondition(n_neighbors, regressor, "KNeighbors")

problem_autosklearn1.add_condition(cond_n_neighbors)

# alpha
alpha = problem_autosklearn1.add_hyperparameter(
    name="alpha", value=(1e-5, 10.0, "log-uniform")
)

cond_alpha = cs.EqualsCondition(alpha, regressor, "MLP")

problem_autosklearn1.add_condition(cond_alpha)

# C
C = problem_autosklearn1.add_hyperparameter(name="C", value=(1e-5, 10.0, "log-uniform"))

cond_C = cs.EqualsCondition(C, regressor, "SVR")

problem_autosklearn1.add_condition(cond_C)

# kernel
kernel = problem_autosklearn1.add_hyperparameter(
    name="kernel", value=["linear", "poly", "rbf", "sigmoid"]
)

cond_kernel = cs.EqualsCondition(kernel, regressor, "SVR")

problem_autosklearn1.add_condition(cond_kernel)

# gamma
gamma = problem_autosklearn1.add_hyperparameter(
    name="gamma", value=(1e-5, 10.0, "log-uniform")
)

cond_gamma = cs.OrConjunction(
    cs.EqualsCondition(gamma, kernel, "rbf"),
    cs.EqualsCondition(gamma, kernel, "poly"),
    cs.EqualsCondition(gamma, kernel, "sigmoid"),
)

problem_autosklearn1.add_condition(cond_gamma)


def run_autosklearn1(config: dict, load_data: callable) -> float:
    """Run function which can be used for AutoML regression.

    It has to be used with the ``deephyper.sklearn.regressor.problem_autosklearn1``  problem definition which corresponds to:

    .. code-block::

        Configuration space object:
            Hyperparameters:
                C, Type: UniformFloat, Range: [1e-05, 10.0], Default: 0.01, on log-scale
                alpha, Type: UniformFloat, Range: [1e-05, 10.0], Default: 0.01, on log-scale
                gamma, Type: UniformFloat, Range: [1e-05, 10.0], Default: 0.01, on log-scale
                kernel, Type: Categorical, Choices: {linear, poly, rbf, sigmoid}, Default: linear
                max_depth, Type: UniformInteger, Range: [2, 100], Default: 14, on log-scale
                n_estimators, Type: UniformInteger, Range: [1, 2000], Default: 45, on log-scale
                n_neighbors, Type: UniformInteger, Range: [1, 100], Default: 50
                regressor, Type: Categorical, Choices: {RandomForest, Linear, AdaBoost, KNeighbors, MLP, SVR, XGBoost}, Default: RandomForest
            Conditions:
                (gamma | kernel == 'rbf' || gamma | kernel == 'poly' || gamma | kernel == 'sigmoid')
                (n_estimators | regressor == 'RandomForest' || n_estimators | regressor == 'AdaBoost')
                C | regressor == 'SVR'
                alpha | regressor == 'MLP'
                kernel | regressor == 'SVR'
                max_depth | regressor == 'RandomForest'
                n_neighbors | regressor == 'KNeighbors'

    Args:
        config (dict): an hyperparameter configuration ``dict`` corresponding to the ``deephyper.sklearn.regressor.problem_autosklearn1``.
        load_data (callable): a function returning data as Numpy arrays ``(X, y)``.

    Returns:
        float: returns the :math:`R^2` on the validation set.
    """
    config["random_state"] = config.get("random_state", 42)
    config["n_jobs"] = config.get("n_jobs", 1)

    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=config["random_state"]
    )

    preproc = minmaxstdscaler()
    X_train = preproc.fit_transform(X_train)
    X_test = preproc.transform(X_test)

    mapping = REGRESSORS

    clf_class = mapping[config["regressor"]]

    # keep parameters possible for the current regressor
    sig = signature(clf_class)
    clf_allowed_params = list(sig.parameters.keys())
    clf_params = {
        k: v
        for k, v in config.items()
        if k in clf_allowed_params and not (v in ["nan", "NA"])
    }

    try:  # good practice to manage the fail value yourself...
        clf = clf_class(**clf_params)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X_train, y_train)

        fit_is_complete = True
    except:  # noqa: E722
        fit_is_complete = False

    if fit_is_complete:
        y_pred = clf.predict(X_test)
        r2 = r2_score(y_test, y_pred)
    else:
        r2 = -1.0

    return r2


if __name__ == "__main__":
    print(problem_autosklearn1)

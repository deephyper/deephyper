"""
This module provides ``problem_autosklearn1`` and ``run_autosklearn`` for classification tasks.
"""
import warnings
from inspect import signature

import ConfigSpace as cs
from deephyper.problem import HpProblem
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier


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


problem_autosklearn1 = HpProblem()

classifier = problem_autosklearn1.add_hyperparameter(
    name="classifier",
    value=[
        "RandomForest",
        "Logistic",
        "AdaBoost",
        "KNeighbors",
        "MLP",
        "SVC",
        "XGBoost",
    ],
)

# n_estimators
n_estimators = problem_autosklearn1.add_hyperparameter(
    name="n_estimators", value=(1, 2000, "log-uniform")
)

cond_n_estimators = cs.OrConjunction(
    cs.EqualsCondition(n_estimators, classifier, "RandomForest"),
    cs.EqualsCondition(n_estimators, classifier, "AdaBoost"),
)

problem_autosklearn1.add_condition(cond_n_estimators)

# max_depth
max_depth = problem_autosklearn1.add_hyperparameter(
    name="max_depth", value=(2, 100, "log-uniform")
)

cond_max_depth = cs.EqualsCondition(max_depth, classifier, "RandomForest")

problem_autosklearn1.add_condition(cond_max_depth)

# n_neighbors
n_neighbors = problem_autosklearn1.add_hyperparameter(
    name="n_neighbors", value=(1, 100)
)

cond_n_neighbors = cs.EqualsCondition(n_neighbors, classifier, "KNeighbors")

problem_autosklearn1.add_condition(cond_n_neighbors)

# alpha
alpha = problem_autosklearn1.add_hyperparameter(
    name="alpha", value=(1e-5, 10.0, "log-uniform")
)

cond_alpha = cs.EqualsCondition(alpha, classifier, "MLP")

problem_autosklearn1.add_condition(cond_alpha)

# C
C = problem_autosklearn1.add_hyperparameter(name="C", value=(1e-5, 10.0, "log-uniform"))

cond_C = cs.OrConjunction(
    cs.EqualsCondition(C, classifier, "Logistic"),
    cs.EqualsCondition(C, classifier, "SVC"),
)

problem_autosklearn1.add_condition(cond_C)

# kernel
kernel = problem_autosklearn1.add_hyperparameter(
    name="kernel", value=["linear", "poly", "rbf", "sigmoid"]
)

cond_kernel = cs.EqualsCondition(kernel, classifier, "SVC")

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


# Mapping available classifiers
CLASSIFIERS = {
    "RandomForest": RandomForestClassifier,
    "Logistic": LogisticRegression,
    "AdaBoost": AdaBoostClassifier,
    "KNeighbors": KNeighborsClassifier,
    "MLP": MLPClassifier,
    "SVC": SVC,
    "XGBoost": XGBClassifier,
}


def run_autosklearn1(config: dict, load_data: callable) -> float:
    """Run function which can be used for AutoML classification.

    It has to be used with the ``deephyper.sklearn.classifier.problem_autosklearn1``  problem definition which corresponds to:

    .. code-block::

        Configuration space object:
            Hyperparameters:
                C, Type: UniformFloat, Range: [1e-05, 10.0], Default: 0.01, on log-scale
                alpha, Type: UniformFloat, Range: [1e-05, 10.0], Default: 0.01, on log-scale
                classifier, Type: Categorical, Choices: {RandomForest, Logistic, AdaBoost, KNeighbors, MLP, SVC, XGBoost}, Default: RandomForest
                gamma, Type: UniformFloat, Range: [1e-05, 10.0], Default: 0.01, on log-scale
                kernel, Type: Categorical, Choices: {linear, poly, rbf, sigmoid}, Default: linear
                max_depth, Type: UniformInteger, Range: [2, 100], Default: 14, on log-scale
                n_estimators, Type: UniformInteger, Range: [1, 2000], Default: 45, on log-scale
                n_neighbors, Type: UniformInteger, Range: [1, 100], Default: 50
            Conditions:
                (C | classifier == 'Logistic' || C | classifier == 'SVC')
                (gamma | kernel == 'rbf' || gamma | kernel == 'poly' || gamma | kernel == 'sigmoid')
                (n_estimators | classifier == 'RandomForest' || n_estimators | classifier == 'AdaBoost')
                alpha | classifier == 'MLP'
                kernel | classifier == 'SVC'
                max_depth | classifier == 'RandomForest'
                n_neighbors | classifier == 'KNeighbors'

    Args:
        config (dict): an hyperparameter configuration ``dict`` corresponding to the ``deephyper.sklearn.classifier.problem_autosklearn1``.
        load_data (callable): a function returning data as Numpy arrays ``(X, y)``.

    Returns:
        float: returns the accuracy on the validation set.
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

    mapping = CLASSIFIERS

    clf_class = mapping[config["classifier"]]

    # keep parameters possible for the current classifier
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
        acc = accuracy_score(y_test, y_pred)
    else:
        acc = -1.0

    return acc


if __name__ == "__main__":
    print(problem_autosklearn1)

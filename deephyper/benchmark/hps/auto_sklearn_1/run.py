import inspect
from inspect import signature
from pprint import pprint

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from load_data import load_data
from problem import Problem


def run(config):
    seed = 42

    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=seed
    )

    mapping = {
        "RandomForest": RandomForestClassifier,
        "Logistic": LogisticRegression,
        "AdaBoost": AdaBoostClassifier,
        "KNeighbors": KNeighborsClassifier,
        "MLP": MLPClassifier,
        "SVC": SVC,
        "XGBoost": XGBClassifier,
    }

    clf_class = mapping[config["classifier"]]

    # keep parameters possible for the current classifier
    sig = signature(clf_class)
    clf_allowed_params = list(sig.parameters.keys())
    clf_params = {
        k: v for k, v in config.items() if k in clf_allowed_params and v != "nan"
    }

    if "n_jobs" in clf_allowed_params:  # performance parameter
        clf_params["n_jobs"] = 8

    try:  # good practice to manage the fail value yourself...
        clf = clf_class(random_state=seed, **clf_params)

        clf.fit(X_train, y_train)

        fit_is_complete = True
    except:
        fit_is_complete = False

    if fit_is_complete:
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
    else:
        acc = -1.0

    return acc

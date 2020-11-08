import inspect
from inspect import signature
from pprint import pprint

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from deephyper.search.hps.automl.classifier.mapping import CLASSIFIERS
from deephyper.nas.preprocessing import minmaxstdscaler


def run(config: dict, load_data: callable) -> float:
    """Run function which can be used for AutoML classification.

    Args:
        config (dict): [description]
        load_data (callable): [description]

    Returns:
        float: [description]
    """
    seed = 42
    config["random_state"] = seed

    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=seed
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

    if "n_jobs" in clf_allowed_params:  # performance parameter
        clf_params["n_jobs"] = 8

    try:  # good practice to manage the fail value yourself...
        clf = clf_class(**clf_params)

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

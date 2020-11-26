from deephyper.problem import filter_parameters
from dhproj.advanced_hpo.mapping import CLASSIFIERS
from dhproj.rf_tuning.load_data import load_data
from sklearn.metrics import accuracy_score
from sklearn.utils import check_random_state


def run(config: dict) -> float:
    """Run function which can be used for AutoML classification.

    Args:
        config (dict): [description]
        load_data (callable): [description]

    Returns:
        float: [description]
    """
    seed = 42
    config["random_state"] = check_random_state(42)

    (X_train, y_train), (X_valid, y_valid) = load_data()

    clf_class = CLASSIFIERS[config["classifier"]]

    # keep parameters possible for the current classifier
    config["n_jobs"] = 4
    clf_params = filter_parameters(clf_class, config)

    try:  # good practice to manage the fail value yourself...
        clf = clf_class(**clf_params)

        clf.fit(X_train, y_train)

        fit_is_complete = True
    except:
        fit_is_complete = False

    if fit_is_complete:
        y_pred = clf.predict(X_valid)
        acc = accuracy_score(y_valid, y_pred)
    else:
        acc = -1.0

    return acc

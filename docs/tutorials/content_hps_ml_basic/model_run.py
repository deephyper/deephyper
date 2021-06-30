from sklearn.utils import check_random_state
from sklearn.ensemble import RandomForestClassifier

from dhproj.rf_tuning.load_data import load_data


def run(config):

    rs = check_random_state(42)

    (X, y), (vX, vy) = load_data()

    classifier = RandomForestClassifier(
        n_jobs=8,
        random_state=rs,
        n_estimators=config["n_estimators"],
        criterion=config["criterion"],
        max_depth=config["max_depth"],
        min_samples_split=config["min_samples_split"]
    )
    classifier.fit(X, y)

    mean_accuracy = classifier.score(vX, vy)

    return mean_accuracy

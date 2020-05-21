from sklearn import preprocessing
import numpy as np

from deephyper.benchmark.datasets import covertype
from deephyper.benchmark.datasets.util import cache_load_data


@cache_load_data("/dev/shm/covertype.npz")
def load_data_cache():

    (X_train, y_train), (X_valid, y_valid), _ = covertype.load_data(seed=42)
    preprocessor = preprocessing.OneHotEncoder()
    y_train = y_train.reshape(-1, 1)
    y_valid = y_valid.reshape(-1, 1)
    y_train = preprocessor.fit_transform(y_train).toarray()
    y_valid = preprocessor.transform(y_valid).toarray()
    print(f"X_train shape: {np.shape(X_train)}")
    print(f"y_train shape: {np.shape(y_train)}")
    print(f"X_valid shape: {np.shape(X_valid)}")
    print(f"y_valid shape: {np.shape(y_valid)}")
    return (X_train, y_train), (X_valid, y_valid)


def test_baseline():
    """Test data with RandomForest

    accuracy_score on Train:  1.0
    accuracy_score on Test:  0.9408463203126216
    balanced_acc on Train:  1.0
    balanced_acc on Test:  0.877023562682862
    """
    from sklearn.ensemble import RandomForestClassifier
    from deephyper.baseline import BaseClassifierPipeline
    from sklearn.utils import class_weight
    from sklearn import metrics

    def load_data():
        train, valid, _ = covertype.load_data(seed=42)
        return train, valid

    train, valid = load_data()
    prop_train = np.bincount(train[1]) / len(train[1])
    prop_valid = np.bincount(valid[1]) / len(valid[1])

    print("classes: ", np.unique(train[1]))
    print("prop_train: ", prop_train)
    print("prop_valid: ", prop_valid)

    baseline_classifier = BaseClassifierPipeline(
        RandomForestClassifier(n_jobs=6), load_data
    )
    baseline_classifier.run()

    def balanced_acc(y_true, y_pred):
        cw = class_weight.compute_class_weight("balanced", np.unique(y_true), y_true)
        sw = np.array([cw[class_ - 1] for class_ in y_true])
        bacc = metrics.accuracy_score(y_true, y_pred, sample_weight=sw)
        return bacc

    baseline_classifier.evaluate(balanced_acc)


def load_data():
    return load_data_cache()


if __name__ == "__main__":
    # load_data()
    test_baseline()

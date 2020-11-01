"""Wrapper around the covertype dataset from Scikit-learn:
https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_covtype.html#sklearn.datasets.fetch_covtype
"""
import numpy as np
from sklearn import datasets
from sklearn import model_selection


def load_data(random_state=42, test_size=0.33, valid_size=0.33):
    # Random State
    random_state = (
        np.random.RandomState(random_state) if type(random_state) is int else random_state
    )

    X, y = datasets.fetch_covtype(return_X_y=True)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=test_size, shuffle=True, random_state=random_state
    )

    # relative valid_size on Train set
    r_valid_size = valid_size / (1.0 - test_size)
    X_train, X_valid, y_train, y_valid = model_selection.train_test_split(
        X_train, y_train, test_size=r_valid_size, shuffle=True, random_state=random_state
    )

    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)


def test_load_data_covertype():
    from deephyper.benchmark.datasets import covertype
    import numpy as np

    names = ["train", "valid", "test"]
    data = covertype.load_data(random_state=42)
    for (X, y), subset_name in zip(data, names):
        print(
            f"X_{subset_name} shape: ",
            np.shape(X),
            f", y_{subset_name} shape: ",
            np.shape(y),
        )


if __name__ == "__main__":
    test_load_data_covertype()

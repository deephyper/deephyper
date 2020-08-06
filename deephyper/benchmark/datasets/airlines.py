"""Wrapper around the covertype dataset from Scikit-learn:
https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_covtype.html#sklearn.datasets.fetch_covtype
"""
import numpy as np
import openml
from openml.datasets.functions import get_dataset
from sklearn import model_selection


def load_data(random_state=42, summary=False):
    # Random State
    random_state = (
        np.random.RandomState(random_state) if type(random_state) is int else random_state
    )

    dataset = openml.datasets.get_dataset(1169)

    if summary:
        # Print a summary
        print(
            f"This is dataset '{dataset.name}', the target feature is "
            f"'{dataset.default_target_attribute}'"
        )
        print(f"URL: {dataset.url}")
        print(dataset.description[:500])

    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="array", target=dataset.default_target_attribute
    )

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.33, shuffle=True, random_state=random_state
    )

    X_train, X_valid, y_train, y_valid = model_selection.train_test_split(
        X_train, y_train, test_size=0.33, shuffle=True, random_state=random_state
    )

    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)


def test_load_data_airlines():
    from deephyper.benchmark.datasets import airlines
    import numpy as np

    names = ["train", "valid", "test "]
    data = airlines.load_data(random_state=42, summary=True)
    for (X, y), subset_name in zip(data, names):
        print(
            f"X_{subset_name} shape: ",
            np.shape(X),
            f", y_{subset_name} shape: ",
            np.shape(y),
        )


if __name__ == "__main__":
    test_load_data_airlines()

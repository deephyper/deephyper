import numpy as np
from sklearn.utils import resample
from deephyper.benchmark.datasets import airlines as dataset


def load_data():

    # In this case passing a random state is critical to make sure
    # that the same data are loaded all the time and that the test set
    # is not mixed with either the training or validation set.
    # It is important to not avoid setting a global seed for safety reasons.
    random_state = np.random.RandomState(seed=42)

    # Proportion of the test set on the full dataset
    ratio_test = 0.33

    # Proportion of the valid set on "dataset \ test set"
    # here we want the test and validation set to have same number of elements
    ratio_valid = (1 - ratio_test) * 0.33

    # The 3rd result is ignored with "_" because it corresponds to the test set
    # which is not interesting for us now.
    (X_train, y_train), (X_valid, y_valid), _, _ = dataset.load_data(
        random_state=random_state, test_size=ratio_test, valid_size=ratio_valid
    )

    X_train, y_train = resample(X_train, y_train, n_samples=int(1e4))
    X_valid, y_valid = resample(X_valid, y_valid, n_samples=int(1e4))

    print(f"X_train shape: {np.shape(X_train)}")
    print(f"y_train shape: {np.shape(y_train)}")
    print(f"X_valid shape: {np.shape(X_valid)}")
    print(f"y_valid shape: {np.shape(y_valid)}")
    return (X_train, y_train), (X_valid, y_valid)


if __name__ == "__main__":
    load_data()

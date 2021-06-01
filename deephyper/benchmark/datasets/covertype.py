import numpy as np
import openml
from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection


def load_data(
    random_state=42,
    verbose=False,
    test_size=0.33,
    valid_size=0.33,
    categoricals_to_integers=False,
):
    """Load the "Covertype" dataset from OpenML.

    Args:
        random_state (int, optional): A numpy `RandomState`. Defaults to 42.
        verbose (bool, optional): Print informations about the dataset. Defaults to False.
        test_size (float, optional): The proportion of the test dataset out of the whole data. Defaults to 0.33.
        valid_size (float, optional): The proportion of the train dataset out of the whole data without the test data. Defaults to 0.33.
        categoricals_to_integers (bool, optional): Convert categoricals features to integer values. Defaults to False.

    Returns:
        tuple: Numpy arrays as, `(X_train, y_train), (X_valid, y_valid), (X_test, y_test)`.
    """
    random_state = (
        np.random.RandomState(random_state) if type(random_state) is int else random_state
    )

    dataset = openml.datasets.get_dataset(150)

    if verbose:
        print(
            f"This is dataset '{dataset.name}', the target feature is "
            f"'{dataset.default_target_attribute}'"
        )
        print(f"URL: {dataset.url}")
        print(dataset.description[:500])

    X, y, categorical_indicator, ft_names = dataset.get_data(
        target=dataset.default_target_attribute
    )

    # encode categoricals as integers
    if categoricals_to_integers:
        for (ft_ind, ft_name) in enumerate(ft_names):
            if categorical_indicator[ft_ind]:
                labenc = LabelEncoder().fit(X[ft_name])
                X[ft_name] = labenc.transform(X[ft_name])
                n_classes = len(labenc.classes_)
            else:
                n_classes = -1
            categorical_indicator[ft_ind] = (
                categorical_indicator[ft_ind],
                n_classes,
            )

    X, y = X.to_numpy(), y.to_numpy()

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=test_size, shuffle=True, random_state=random_state
    )

    # relative valid_size on Train set
    r_valid_size = valid_size / (1.0 - test_size)
    X_train, X_valid, y_train, y_valid = model_selection.train_test_split(
        X_train, y_train, test_size=r_valid_size, shuffle=True, random_state=random_state
    )

    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test), categorical_indicator


def test_load_data_covertype():
    from deephyper.benchmark.datasets import covertype
    import numpy as np

    names = ["train", "valid", "test"]
    data = covertype.load_data(random_state=42, verbose=True)[:-1]
    for (X, y), subset_name in zip(data, names):
        print(
            f"X_{subset_name} shape: ",
            np.shape(X),
            f", y_{subset_name} shape: ",
            np.shape(y),
        )


if __name__ == "__main__":
    test_load_data_covertype()

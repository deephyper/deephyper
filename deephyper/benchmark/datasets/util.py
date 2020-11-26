import logging
import os

import numpy as np
import openml
from sklearn import model_selection

logger = logging.getLogger(__name__)


def cache_load_data(cache_loc):
    """Decorator of load_data function to dache numpy arrays return by the function. The load_data function should return a tuple of the form: ``(X_train, y_train), (X_valid, y_valid)``.

    Args:
        cache_loc (str): path where the data will be cached.
    """

    def _cache(data_loader):
        def wrapper(*args, **kwargs):
            if os.path.exists(cache_loc):
                logger.info("Reading data from cache")
                with open(cache_loc, "rb") as fp:
                    data = {k: arr for k, arr in np.load(fp).items()}
                return (
                    (data["X_train"], data["y_train"]),
                    (data["X_valid"], data["y_valid"]),
                )

            else:
                (X_train, y_train), (X_valid, y_valid) = data_loader(*args, **kwargs)
                if os.path.exists(os.path.dirname(cache_loc)):
                    logger.info("Data not cached; invoking user data loader")
                    data = {
                        "X_train": X_train,
                        "y_train": y_train,
                        "X_valid": X_valid,
                        "y_valid": y_valid,
                    }
                    with open(cache_loc, "wb") as fp:
                        np.savez(fp, **data)
                else:
                    logger.warning(
                        "Data cannot be cached because the path does not exist. Returning data anyway."
                    )
                return (X_train, y_train), (X_valid, y_valid)

        return wrapper

    return _cache


def get_openml_dataset(
    dataset_id, random_state=42, summary=False, test_size=0.33, valid_size=0.33
):
    # Random State
    random_state = (
        np.random.RandomState(random_state) if type(random_state) is int else random_state
    )

    dataset = openml.datasets.get_dataset(dataset_id)

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
        X, y, test_size=test_size, shuffle=True, random_state=random_state
    )

    # relative valid_size on Train set
    r_valid_size = valid_size / (1.0 - test_size)
    X_train, X_valid, y_train, y_valid = model_selection.train_test_split(
        X_train, y_train, test_size=r_valid_size, shuffle=True, random_state=random_state
    )

    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)

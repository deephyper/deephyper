import logging
import os
import numpy as np

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

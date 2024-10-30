import numpy as np


def average(x: np.ndarray | np.ma.MaskedArray, axis=None, weights=None):
    """Check if ``x`` is a classic numpy array or a masked array to apply the corresponding
    implementation.

    Args:
        x (np.ndarray | np.ma.MaskedArray): array like.
        axis (int, optional): the axis. Defaults to ``None``.
        weights (list, optional): the weights. Defaults to ``None``.

    Returns:
        array like: the average.
    """
    numpy_func = np.average
    if isinstance(x, np.ma.MaskedArray):
        numpy_func = np.ma.average
    return numpy_func(x, axis=axis, weights=weights)

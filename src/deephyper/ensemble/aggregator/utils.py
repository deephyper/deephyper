import numpy as np
from typing import Optional, Union, Sequence


def average(
    x: Union[np.ndarray, np.ma.MaskedArray],
    axis: Optional[int] = None,
    weights: Optional[Sequence[float]] = None,
) -> Union[np.ndarray, np.ma.MaskedArray]:
    """Compute the weighted average of an array, supporting both NumPy arrays and masked arrays.

    Args:
        x (np.ndarray | np.ma.MaskedArray): Input array or masked array.
        axis (int, optional): Axis along which to compute the average. Defaults to None.
        weights (Sequence[float], optional): Weights to apply. Defaults to None.

    Returns:
        np.ndarray | np.ma.MaskedArray: Weighted average of the input array.

    Raises:
        TypeError: If `x` is not a NumPy array or masked array.
    """
    if not isinstance(x, (np.ndarray, np.ma.MaskedArray)):
        raise TypeError("Input `x` must be a numpy.ndarray or numpy.ma.MaskedArray.")

    numpy_func = np.ma.average if isinstance(x, np.ma.MaskedArray) else np.average
    return numpy_func(x, axis=axis, weights=weights)

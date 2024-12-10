import numpy as np

from scipy.stats import rankdata


def rank(
    a,
    method="min",
    decimals=3,
    *,
    axis=None,
    nan_policy="propagate",
):
    """Returns the ranking from a list of scores given a tolerance epsilon.

    This function is a wrapper around ``scipy.stats.rankdata``, see `Scipy Documentation
    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rankdata.html>`_).

    Lower scores corresponds to lower ranks.

    Args:
        a (array): List of scores.
        method (str, optional): The method used to assign ranks to tied elements. The options are
            ``"average"``, ``"min"``, ``"max"``, ``"dense"`` and ``'ordinal'``. Defaults to
            ``"min"``.
        decimals (int, optional): The number of decimal at which rounding is performed. Defaults to
            ``3``.
        axis (int, optional): The axis along which the elements of ``a`` are ranked. Defaults to
            ``None`` to rank the elements after flattening the array.
        nan_policy (str, optional): Defines how to handle when input contains nan. The options are
            ``"propagate"``, ``"raise"``, ``"omit"``. Defaults to ``"propagate"``.

    Returns:
        array: The ranking of the scores.
    """
    a = np.array(a).astype(float)
    if decimals is not None:
        rounded_a = np.round(a, decimals=decimals)
    else:
        rounded_a = a
    ranking = rankdata(rounded_a, method=method, axis=axis, nan_policy=nan_policy)
    return ranking

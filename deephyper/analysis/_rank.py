import numpy as np


def rank(scores, decimals: int = 3):
    """Returns the ranking from a list of scores given a tolerance epsilon.

    Args:
        scores (list): List of scores.
        decimals (int, optional): The number of decimal to keep. Defaults to ``3``.
    """
    scores = np.array(scores).astype(float)
    if decimals is not None:
        rounded_scores = np.round(scores, decimals=decimals)
    else:
        rounded_scores = scores
    sorted_idx = np.argsort(rounded_scores)
    sorted_scores = rounded_scores[sorted_idx]
    ranking = np.searchsorted(sorted_scores, rounded_scores) + 1
    return ranking

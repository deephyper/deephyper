import numpy as np


def is_pareto_efficient(new_obj, objvals):
    """Check if the new objective vector is pareto efficient with respect to previously computed values.

    Args:
        new_obj (array or list): Array or list of size (n_objectives, )
        objvals (array or list): Array or list of size (n_points, n_objectives)

    Returns:
        bool: True if the vector is pareto efficient and false otherwise.
    """
    return np.all(np.any(np.asarray(new_obj) < objvals, axis=1))


def pareto_front(y):
    """Extract the pareto front (actual objective values of the non-dominated set).

    Args:
        y (array or list): Array or list of size (n_points, n_objectives)

    Returns:
        array: Subarray of y representing the pareto front.
    """
    nds = non_dominated_set(y, return_mask=False)
    return y[nds]


def non_dominated_set_ranked(y, fraction, return_mask=True):
    """Find the set of top-``fraction x 100%`` of non-dominated points. The number of points returned is ``min(n_points, ceil(fraction * n_points))`` where ``n_points`` is the number of points in the input array. Function assumes minimization.

    Args:
        y (array or list): Array or list of size (n_points, n_objectives)
        fraction (float or int): Fraction of points to return.
        return_mask (bool, optional): Whether to return a mask or the actual indices of the non-dominated set. Defaults to True.

    Raises:
        ValueError: Raised if ``fraction`` is not a non-negative number or if y is not an array of size (n_points, n_objectives).

    Returns:
        array: If return_mask is True, this will be an (n_points, ) boolean array. Else it will be a 1-d integer array of indices indicating which points are in the top non-dominated set.
    """
    if not isinstance(fraction, (float, int)) or fraction < 0:
        raise ValueError("Expected 'fraction' to be a non-negative scalar")
    if np.ndim(y) == 0:
        return np.asarray([fraction > 0.0])

    n_points = np.shape(y)[0]
    req_number = min(np.ceil(fraction * n_points).astype(int), n_points)
    if req_number <= 0:
        return np.zeros(n_points, dtype=bool)
    if req_number >= n_points:
        return np.ones(n_points, dtype=bool)

    chosen_indices = []
    map_indices = np.arange(n_points)
    while len(chosen_indices) < req_number:
        nds = non_dominated_set(y, return_mask=True)
        chosen_indices.extend(map_indices[nds])
        if len(chosen_indices) > req_number:
            del chosen_indices[req_number:]
            break
        y = y[~nds]
        map_indices = map_indices[~nds]
    if return_mask:
        chosen = np.zeros(n_points, dtype=bool)
        chosen[chosen_indices] = True
        return chosen
    return chosen_indices


def non_dominated_set(y, return_mask=True):
    """Find the set of non-dominated points. If there are multiple duplicate non-dominated points, then only one will be included. The function assumes minimization and is adapted from https://stackoverflow.com/a/40239615 by adding a sorting step to improve efficiency.

    Args:
        y (array or list): Array or list of size (n_points, n_objectives)
        return_mask (bool, optional): Whether to return a mask or the actual indices of the non-dominated set. Defaults to True.

    Returns:
        array: If return_mask is True, this will be an (n_points, ) boolean array. Else it will be a 1-d integer array of indices indicating which points are non-dominated.
    """
    if np.ndim(y) == 1:
        y = np.asarray_chkfinite(y)[:, np.newaxis]
    elif np.ndim(y) == 2:
        y = np.asarray_chkfinite(y)
    else:
        raise ValueError("Expected y to be an array of size (n_points, n_objectives)")

    order = np.argsort(y.sum(axis=1))
    costs = y[order]

    n_points = y.shape[0]
    is_efficient = np.arange(n_points)
    idx = 0
    while idx < len(costs):
        nondominated_point_mask = np.any(costs < costs[idx], axis=1)
        nondominated_point_mask[idx] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        idx = np.sum(nondominated_point_mask[:idx], dtype=int) + 1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype=bool)
        is_efficient_mask[is_efficient] = True
        is_efficient_mask[order] = is_efficient_mask.copy()
        return is_efficient_mask

    return order[is_efficient]


def non_dominated_set_dumb(y):
    """For debugging purposes."""
    is_efficient = np.ones(y.shape[0], dtype=bool)
    for i, c in enumerate(y):
        is_efficient[i] = np.all(np.any(y[:i] > c, axis=1)) and np.all(
            np.any(y[i + 1 :] > c, axis=1)
        )
    return is_efficient


if __name__ == "__main__":
    rng = np.random.RandomState(42)
    npoints = 1000
    nobj = 2
    for it in range(100):
        y = rng.rand(npoints, nobj)
        pf = non_dominated_set(y, return_mask=True)
        assert np.array_equal(non_dominated_set_dumb(y), pf)
        assert np.array_equal(
            pf.nonzero()[0], np.sort(non_dominated_set(y, return_mask=False))
        )

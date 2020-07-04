import numpy as np

def organize_data(X, E, MAX_ATOM, N_FEAT, E_FEAT):
    """
    Zero padding node features, adjacency matrix, edge features
    Args:
        X: node features
        E: edge features

    Returns:
        node features, adjacency matrix, edge features
    """
    A = E[..., :6].sum(axis=-1) != 0
    A = A.astype(np.float32)
    X_0 = np.zeros(shape=(MAX_ATOM, N_FEAT))
    X_0[:X.shape[0], :X.shape[1]] = X
    A_0 = np.zeros(shape=(MAX_ATOM, MAX_ATOM))
    A_0[:A.shape[0], :A.shape[1]] = A
    E_0 = np.zeros(shape=(MAX_ATOM, MAX_ATOM, E_FEAT))
    E_0[:E.shape[0], :E.shape[1], :] = E
    return X_0, A_0, E_0
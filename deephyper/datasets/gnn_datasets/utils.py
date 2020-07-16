import numpy as np
import scipy.sparse as sp

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
    A_0 = A_0.reshape(MAX_ATOM * MAX_ATOM, 1)
    E_0 = E_0.reshape(MAX_ATOM * MAX_ATOM, E_FEAT)
    return X_0, A_0, E_0


def organize_data_sparse(X, E, MAX_ATOM, MAX_EDGE, N_FEAT, E_FEAT):
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
    A_0 = np.ones(shape=(MAX_EDGE, 2))*(MAX_ATOM-1)
    A = sp.coo_matrix(A)
    n_edge = len(A.row)
    A_0[:n_edge, 0] = A.row
    A_0[:n_edge, 1] = A.col
    E_0 = np.zeros(shape=(MAX_EDGE, E_FEAT))
    E_0[:n_edge, :] = [e[a.row, a.col] for e, a in zip([E], [A])][0]
    M_0 = np.zeros(shape=(MAX_ATOM, ))
    M_0[:X.shape[0]] = 1
    return X_0, A_0, E_0, M_0

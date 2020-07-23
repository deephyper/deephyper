import numpy as np
import scipy.sparse as sp


# def organize_data(X, E, MAX_ATOM, N_FEAT, E_FEAT):
#     """
#     Zero padding node features, adjacency matrix, edge features
#     Args:
#         X: node features
#         E: edge features
#
#     Returns:
#         node features, adjacency matrix, edge features
#     """
#     A = E[..., :6].sum(axis=-1) != 0
#     A = A.astype(np.float32)
#     X_0 = np.zeros(shape=(MAX_ATOM, N_FEAT))
#     X_0[:X.shape[0], :X.shape[1]] = X
#     A_0 = np.zeros(shape=(MAX_ATOM, MAX_ATOM))
#     A_0[:A.shape[0], :A.shape[1]] = A
#     E_0 = np.zeros(shape=(MAX_ATOM, MAX_ATOM, E_FEAT))
#     E_0[:E.shape[0], :E.shape[1], :] = E
#     A_0 = A_0.reshape(MAX_ATOM * MAX_ATOM, 1)
#     E_0 = E_0.reshape(MAX_ATOM * MAX_ATOM, E_FEAT)
#     return X_0, A_0, E_0


def organize_data_sparse(X, E, MAX_ATOM, MAX_EDGE, N_FEAT, E_FEAT):
    """
    Zero padding node features, adjacency matrix, edge features
    Args:
        X: node features
        E: edge features
        MAX_ATOM: the maximum number of atoms zero-padding to
        MAX_EDGE: the maximum number of edges zero-padding to
        N_FEAT: the number of node features
        E_FEAT: the number of edge features
    Returns:
        X_0: the node features (batch * MAX_ATOM * N_FEAT)
        A_0: the edge pairs (batch * MAX_EDGE * 2)
        E_0: the edge features (batch * MAX_EDGE * E_FEAT)
        M_0: the mask of actual atoms (batch * MAX_ATOM)
        N_0: the inverse sqrt of node degrees for GCN attention 1/sqrt(N(i)*N(j)) (batch * MAX_EDGE)
    """
    # The adjacency matrix A
    A = E[..., :6].sum(axis=-1) != 0
    A = A.astype(np.float32)

    # The node feature X_0
    X_0 = np.zeros(shape=(MAX_ATOM, N_FEAT))
    X_0[:X.shape[0], :X.shape[1]] = X

    # Convert A to edge pair format (if I use A_0 = np.zeros(...), the 0 to 0 pair will be emphasized a lot)
    # So I set all to the max_atom, then the max_atom atom has no node features.
    # And I mask all calculation for existing atoms.
    A_0 = np.ones(shape=(MAX_EDGE + MAX_ATOM, 2)) * (MAX_ATOM - 1)
    A = sp.coo_matrix(A)
    n_edge = len(A.row)
    A_0[:n_edge, 0] = A.row
    A_0[:n_edge, 1] = A.col

    # The edge feature E_0
    E_0 = np.zeros(shape=(MAX_EDGE + MAX_ATOM, E_FEAT))
    E_0[:n_edge, :] = [e[a.row, a.col] for e, a in zip([E], [A])][0]

    # Fill the zeros in A_0 with self loop
    A_0[MAX_EDGE:, 0] = np.array([i for i in range(MAX_ATOM)])
    A_0[MAX_EDGE:, 1] = np.array([i for i in range(MAX_ATOM)])

    # The mask for existing nodes
    M_0 = np.zeros(shape=(MAX_ATOM,))
    M_0[:X.shape[0]] = 1

    # The inverse of sqrt of node degrees
    outputa = np.unique(A_0[:, 0], return_counts=True, return_inverse=True)
    outputb = np.unique(A_0[:, 1], return_counts=True, return_inverse=True)
    n_a = []
    for element in outputa[1]:
        n_a.append(outputa[2][element])
    n_b = []
    for element in outputb[1]:
        n_b.append(outputb[2][element])
    n_a = np.array(n_a)
    n_b = np.array(n_b)
    n_0 = np.multiply(n_a, n_b)
    N_0 = 1 / np.sqrt(n_0)

    return X_0, A_0, E_0, M_0, N_0


def load_molnet_single(data_func, MAX_ATOM_EDGE, split='stratified', seed=0):
    """
    Load a single molenet dataset
    Args:
        data_func: func, the function to load a single dataset
        MAX_ATOM_EDGE: list, contains the max atom and max edge
        split: str, the split type
        seed: int, the seed for dataset split

    Returns:
        Training, validation and testing data, and task name and transformers.
    """
    MAX_ATOM = MAX_ATOM_EDGE[0] + 1
    MAX_EDGE = MAX_ATOM_EDGE[1] + 1
    N_FEAT = 75
    E_FEAT = 14

    task_name, (train_dataset, valid_dataset, test_dataset), transformers = data_func(featurizer='Weave',
                                                                                      split=split,
                                                                                      move_mean=True,
                                                                                      seed=seed)
    X_train, X_valid, X_test = [], [], []
    A_train, A_valid, A_test = [], [], []
    E_train, E_valid, E_test = [], [], []
    y_train, y_valid, y_test = [], [], []
    M_train, M_valid, M_test = [], [], []
    N_train, N_valid, N_test = [], [], []
    train_x = train_dataset.X
    train_y = train_dataset.y
    valid_x = valid_dataset.X
    valid_y = valid_dataset.y
    test_x = test_dataset.X
    test_y = test_dataset.y

    # TRAINING DATASET
    for i in range(len(train_dataset)):
        X, A, E, M, N = organize_data_sparse(train_x[i].nodes, train_x[i].pairs, MAX_ATOM, MAX_EDGE, N_FEAT, E_FEAT)
        X_train.append(X)
        E_train.append(E)
        A_train.append(A)
        M_train.append(M)
        N_train.append(N)
        y_train.append(train_y[i])

    # VALIDATION DATASET
    for i in range(len(valid_dataset)):
        X, A, E, M, N = organize_data_sparse(valid_x[i].nodes, valid_x[i].pairs, MAX_ATOM, MAX_EDGE, N_FEAT, E_FEAT)
        X_valid.append(X)
        E_valid.append(E)
        A_valid.append(A)
        M_valid.append(M)
        N_valid.append(N)
        y_valid.append(valid_y[i])

    # TESTING DATASET
    for i in range(len(test_dataset)):
        X, A, E, M, N = organize_data_sparse(test_x[i].nodes, test_x[i].pairs, MAX_ATOM, MAX_EDGE, N_FEAT, E_FEAT)
        X_test.append(X)
        E_test.append(E)
        A_test.append(A)
        M_test.append(M)
        N_test.append(N)
        y_test.append(test_y[i])
    X_train = np.array(X_train)
    A_train = np.array(A_train)
    E_train = np.array(E_train)
    M_train = np.array(M_train)
    N_train = np.array(N_train)
    y_train = np.array(y_train).squeeze()
    X_valid = np.array(X_valid)
    A_valid = np.array(A_valid)
    E_valid = np.array(E_valid)
    M_valid = np.array(M_valid)
    N_valid = np.array(N_valid)
    y_valid = np.array(y_valid).squeeze()
    X_test = np.array(X_test)
    A_test = np.array(A_test)
    E_test = np.array(E_test)
    M_test = np.array(M_test)
    N_test = np.array(N_test)
    y_test = np.array(y_test).squeeze()
    return [X_train, A_train, E_train, M_train, N_train, y_train], \
           [X_valid, A_valid, E_valid, M_valid, N_valid, y_valid], \
           [X_test, A_test, E_test, M_test, N_test, y_test], \
           task_name, transformers

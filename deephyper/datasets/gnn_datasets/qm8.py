import numpy as np
from deepchem.molnet import load_qm8

# FIXED PARAMETERS

MAX_ATOM = 26
N_FEAT = 70
E_FEAT = 8


def adjacency_list_to_array(adjacency_list):
    """
    Function to convert adjacency list to array
    Args:
        adjacency_list: list containing atom's connection

    Returns:
        Adjacency array
    """
    max_size = len(adjacency_list)
    A = np.zeros(shape=(max_size, max_size))
    for i in range(max_size):
        A[i, adjacency_list[i]] = 1
        A[i, i] = 1
    return A


def organize_data(X, E):
    """
    Zero padding node features, adjacency matrix, edge features
    Args:
        X: node features
        E: edge features

    Returns:
        node features, adjacency matrix, edge features
    """
    A = E[..., :5].sum(axis=-1)
    X_0 = np.zeros(shape=(MAX_ATOM, N_FEAT))
    X_0[:X.shape[0], :X.shape[1]] = X
    A_0 = np.zeros(shape=(MAX_ATOM, MAX_ATOM))
    A_0[:A.shape[0], :A.shape[1]] = A
    E_0 = np.zeros(shape=(MAX_ATOM, MAX_ATOM, E_FEAT))
    E_0[:E.shape[0], :E.shape[1], :] = E
    return X_0, A_0, E_0


def load_qm8_MPNN(split='random'):
    print("Loading qm8 Dataset")
    qm8_tasks, (train_dataset, valid_dataset, test_dataset), transformers = load_qm8(featurizer='MP',
                                                                                     split=split,
                                                                                     move_mean=True)
    X_train, X_valid, X_test = [], [], []
    A_train, A_valid, A_test = [], [], []
    E_train, E_valid, E_test = [], [], []
    y_train, y_valid, y_test = [], [], []
    train_x = train_dataset.X
    train_y = train_dataset.y
    valid_x = valid_dataset.X
    valid_y = valid_dataset.y
    test_x = test_dataset.X
    test_y = test_dataset.y

    # TRAINING DATASET
    for i in range(len(train_dataset)):
        X, A, E = organize_data(train_x[i].nodes, train_x[i].pairs)
        X_train.append(X)
        E_train.append(E)
        A_train.append(A)
        y_train.append(train_y[i])

    # VALIDATION DATASET
    for i in range(len(valid_dataset)):
        X, A, E = organize_data(valid_x[i].nodes, valid_x[i].pairs)
        X_valid.append(X)
        E_valid.append(E)
        A_valid.append(A)
        y_valid.append(valid_y[i])

    # TESTING DATASET
    for i in range(len(test_dataset)):
        X, A, E = organize_data(test_x[i].nodes, test_x[i].pairs)
        X_test.append(X)
        E_test.append(E)
        A_test.append(A)
        y_test.append(test_y[i])
    X_train = np.array(X_train)
    A_train = np.array(A_train)
    E_train = np.array(E_train)
    y_train = np.array(y_train).squeeze()
    X_valid = np.array(X_valid)
    A_valid = np.array(A_valid)
    E_valid = np.array(E_valid)
    y_valid = np.array(y_valid).squeeze()
    X_test = np.array(X_test)
    A_test = np.array(A_test)
    E_test = np.array(E_test)
    y_test = np.array(y_test).squeeze()
    print("Loading qm8 Dataset Finished")
    return [X_train, A_train, E_train, y_train], \
           [X_valid, A_valid, E_valid, y_valid], \
           [X_test, A_test, E_test, y_test], \
           qm8_tasks, transformers


if __name__ == '__main__':
    train_data, valid_data, test_data, qm8_tasks, transformers = load_qm8_MPNN()

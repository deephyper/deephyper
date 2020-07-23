import numpy as np
from deepchem.molnet import load_sampl
from .utils import organize_data, organize_data_sparse

# FIXED PARAMETERS

MAX_ATOM = 24 + 1
MAX_EDGE = 25 + 1
N_FEAT = 75
E_FEAT = 14

#
# def load_freesolv_MPNN(split='random', seed=2020):
#     print("Loading freesolv Dataset")
#     freesolv_tasks, (train_dataset, valid_dataset, test_dataset), transformers = load_sampl(featurizer='Weave',
#                                                                                             split=split,
#                                                                                             move_mean=True)
#     X_train, X_valid, X_test = [], [], []
#     A_train, A_valid, A_test = [], [], []
#     E_train, E_valid, E_test = [], [], []
#     y_train, y_valid, y_test = [], [], []
#     train_x = train_dataset.X
#     train_y = train_dataset.y
#     valid_x = valid_dataset.X
#     valid_y = valid_dataset.y
#     test_x = test_dataset.X
#     test_y = test_dataset.y
#
#     # TRAINING DATASET
#     for i in range(len(train_dataset)):
#         X, A, E = organize_data(train_x[i].nodes, train_x[i].pairs, MAX_ATOM, N_FEAT, E_FEAT)
#         X_train.append(X)
#         E_train.append(E)
#         A_train.append(A)
#         y_train.append(train_y[i])
#
#     # VALIDATION DATASET
#     for i in range(len(valid_dataset)):
#         X, A, E = organize_data(valid_x[i].nodes, valid_x[i].pairs, MAX_ATOM, N_FEAT, E_FEAT)
#         X_valid.append(X)
#         E_valid.append(E)
#         A_valid.append(A)
#         y_valid.append(valid_y[i])
#
#     # TESTING DATASET
#     for i in range(len(test_dataset)):
#         X, A, E = organize_data(test_x[i].nodes, test_x[i].pairs, MAX_ATOM, N_FEAT, E_FEAT)
#         X_test.append(X)
#         E_test.append(E)
#         A_test.append(A)
#         y_test.append(test_y[i])
#     X_train = np.array(X_train)
#     A_train = np.array(A_train)
#     E_train = np.array(E_train)
#     y_train = np.array(y_train).squeeze()
#     X_valid = np.array(X_valid)
#     A_valid = np.array(A_valid)
#     E_valid = np.array(E_valid)
#     y_valid = np.array(y_valid).squeeze()
#     X_test = np.array(X_test)
#     A_test = np.array(A_test)
#     E_test = np.array(E_test)
#     y_test = np.array(y_test).squeeze()
#     print("Loading freesolv Dataset Finished")
#     return [X_train, A_train, E_train, y_train], \
#            [X_valid, A_valid, E_valid, y_valid], \
#            [X_test, A_test, E_test, y_test], \
#            freesolv_tasks, transformers


def load_freesolv_MPNN_sparse(split='random', seed=2020):
    print("Loading freesolv Dataset")
    freesolv_tasks, (train_dataset, valid_dataset, test_dataset), transformers = load_sampl(featurizer='Weave',
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
    print("Loading freesolv Dataset Finished")
    return [X_train, A_train, E_train, M_train, N_train, y_train], \
           [X_valid, A_valid, E_valid, M_valid, N_valid, y_valid], \
           [X_test, A_test, E_test, M_test, N_test, y_test], \
           freesolv_tasks, transformers


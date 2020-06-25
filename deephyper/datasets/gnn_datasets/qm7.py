import numpy as np
from deepchem.molnet import load_qm7_from_mat


def adjacency_list_to_array(adjacency_list):
    """
    Function to convert adjacency list to array
    Args:
        adjacency_list: list containing atom's connection

    Returns:
        Adjacency array
    """
    max_size = len(adjacency_list)
    adjacency_array = np.zeros(shape=(max_size, max_size))
    for i in range(max_size):
        adjacency_array[i, adjacency_list[i]] = 1
        adjacency_array[i, i] = 1
    return adjacency_array


def load_qm7(zero_padding=True, split='stratified'):
    """
    Load qm7 dataset from .mat file. Max atom 23.
    Check details here: http://quantum-machine.org/datasets/
    Args:
        zero_padding: bool, by default True. Padding node feature array X to (MAX_ATOM, N_FEAT), adjacency array A to
        (MAX_ATOM, MAX_ATOM);
        split: str, {'index', 'random', 'stratified', None};
    Returns:
        Lists of train, valid and test data, and qm7 task name.
    """
    print("Loading QM7 Dataset (takes about 1 minute)...")
    qm7_tasks, (train_dataset, valid_dataset, test_dataset), transformers = load_qm7_from_mat(featurizer='GraphConv',
                                                                                              split=split,
                                                                                              move_mean=True)
    MAX_ATOM = 23
    N_FEAT = 75
    X_train, X_valid, X_test = [], [], []
    A_train, A_valid, A_test = [], [], []
    y_train, y_valid, y_test = [], [], []
    train_x = train_dataset.X
    train_y = train_dataset.y
    valid_x = valid_dataset.X
    valid_y = valid_dataset.y
    test_x = test_dataset.X
    test_y = test_dataset.y

    # TRAINING DATASET
    for i in range(len(train_dataset)):
        atom_features = train_x[i].atom_features
        adjacency_list = train_x[i].get_adjacency_list()
        adjacency_array = adjacency_list_to_array(adjacency_list)

        if zero_padding:
            atom_features_zero_padding = np.zeros(shape=(MAX_ATOM, N_FEAT))
            atom_features_zero_padding[:atom_features.shape[0], :atom_features.shape[1]] = atom_features
            adjacency_array_zero_padding = np.zeros(shape=(MAX_ATOM, MAX_ATOM))
            adjacency_array_zero_padding[:adjacency_array.shape[0], :adjacency_array.shape[1]] = adjacency_array
            X_train.append(atom_features_zero_padding)
            A_train.append(adjacency_array_zero_padding)
        else:
            X_train.append(atom_features)
            A_train.append(adjacency_array)
        y_train.append(train_y[i])

    # VALIDATION DATASET
    for i in range(len(valid_dataset)):
        atom_features = valid_x[i].atom_features
        adjacency_list = valid_x[i].get_adjacency_list()
        adjacency_array = adjacency_list_to_array(adjacency_list)

        if zero_padding:
            atom_features_zero_padding = np.zeros(shape=(MAX_ATOM, N_FEAT))
            atom_features_zero_padding[:atom_features.shape[0], :atom_features.shape[1]] = atom_features
            adjacency_array_zero_padding = np.zeros(shape=(MAX_ATOM, MAX_ATOM))
            adjacency_array_zero_padding[:adjacency_array.shape[0], :adjacency_array.shape[1]] = adjacency_array
            X_valid.append(atom_features_zero_padding)
            A_valid.append(adjacency_array_zero_padding)
        else:
            X_valid.append(atom_features)
            A_valid.append(adjacency_array)
        y_valid.append(valid_y[i])

    # TESTING DATASET
    for i in range(len(test_dataset)):
        atom_features = test_x[i].atom_features
        adjacency_list = test_x[i].get_adjacency_list()
        adjacency_array = adjacency_list_to_array(adjacency_list)

        if zero_padding:
            atom_features_zero_padding = np.zeros(shape=(MAX_ATOM, N_FEAT))
            atom_features_zero_padding[:atom_features.shape[0], :atom_features.shape[1]] = atom_features
            adjacency_array_zero_padding = np.zeros(shape=(MAX_ATOM, MAX_ATOM))
            adjacency_array_zero_padding[:adjacency_array.shape[0], :adjacency_array.shape[1]] = adjacency_array
            X_test.append(atom_features_zero_padding)
            A_test.append(adjacency_array_zero_padding)
        else:
            X_test.append(atom_features)
            A_test.append(adjacency_array)
        y_test.append(test_y[i])

    return [X_train, A_train, y_train], \
           [X_valid, A_valid, y_valid], \
           [X_test, A_test, y_test], \
           qm7_tasks


if __name__ == '__main__':
    train_data, valid_data, test_data, qm7_tasks = load_data()

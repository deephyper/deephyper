import numpy as np
from deepchem.molnet import load_qm7_from_mat, load_qm8, load_qm9, load_delaney, load_sampl, load_lipo
from .utils import load_molnet_single


def load_molnet(data='qm7', split='stratified', seed=0):
    """
    load a certain molecule-net dataset
    Args:
        data: str, the name of the dataset
        split: str, the type of split
        seed: int, the seed to split dataset

    Returns:

    """
    assert data in ['qm7', 'qm8', 'qm9', 'esol', 'freesolv', 'lipo'], "Dataset not included."
    if data == 'qm7':
        MAX_ATOM_EDGE = [8, 10]
        data_func = load_qm7_from_mat
    elif data == 'qm8':
        MAX_ATOM_EDGE = [9, 14]
        data_func = load_qm8
    elif data == 'qm9':
        MAX_ATOM_EDGE = [9, 16]
        data_func = load_qm9
    elif data == 'esol':
        MAX_ATOM_EDGE = [55, 68]
        data_func = load_delaney
    elif data == 'freesolv':
        MAX_ATOM_EDGE = [24, 25]
        data_func = load_sampl
    elif data == 'lipo':
        MAX_ATOM_EDGE = [115, 236]
        data_func = load_lipo
    [X_train, A_train, E_train, M_train, N_train, y_train], \
    [X_valid, A_valid, E_valid, M_valid, N_valid, y_valid], \
    [X_test, A_test, E_test, M_test, N_test, y_test], \
    task_name, transformers = load_molnet_single(data_func, MAX_ATOM_EDGE, split, seed)
    return [X_train, A_train, E_train, M_train, N_train, y_train], \
           [X_valid, A_valid, E_valid, M_valid, N_valid, y_valid], \
           [X_test, A_test, E_test, M_test, N_test, y_test], \
           task_name, transformers


def test_load_molnet():
    load_molnet(data='qm7', split='stratified', seed=0)
    load_molnet(data='qm8', split='random', seed=0)
    load_molnet(data='qm9', split='random', seed=0)
    load_molnet(data='esol', split='random', seed=0)
    load_molnet(data='freesolv', split='random', seed=0)
    load_molnet(data='lipo', split='random', seed=0)


if __name__ == '__main__':
    test_load_molnet()


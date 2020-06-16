import os
import numpy as np
import pickle
# from protonation.gnnproton.Data_Processing import molecule
# from protonation.gnnproton.train_utils import load_data_split, load_dict, load_data_func


def load_data():
    # define necessary parameters
    elem_list = ['C', 'O', 'H', 'unknown']  # species of atoms
    atom_fdim = len(elem_list) + 6 + 6 + 6 + 1  # number of atomic features
    bond_fdim = 6  # number of bond features
    max_nb = 6  # maximum number of neighbors
    output_fdim = 1  # output feature dimension
    max_atom = 31  # maximum atoms in the molecule

    # define directories
    if os.path.exists(r"/mnt/"):
        DATA_DIR = r"/mnt/d/machinelearning2/ANL/archive/gnn/Data/Train_Data/Total_Data.pkl"
        TRAIN_DIR = r"/mnt/d/machinelearning2/ANL/archive/gnn/Data/Train_Data/Train_Keys.txt"
        VALID_DIR = r"/mnt/d/machinelearning2/ANL/archive/gnn/Data/Train_Data/Valid_Keys.txt"
        TEST_DATA_DIR = r"/mnt/d/machinelearning2/ANL/archive/gnn/Data/Test_Data/Test_Data.pkl"
        TEST_DIR = r"/mnt/d/machinelearning2/ANL/archive/gnn/Data/Test_Data/Test_Keys.txt"
    elif os.path.exists(r"D:\\machinelearning2\\"):
        DATA_DIR = r"D:\\machinelearning2\\ANL\\archive\\gnn\\Data\\Train_Data\\Total_Data.pkl"
        TRAIN_DIR = r"D:\\machinelearning2\\ANL\\archive\\gnn\\Data\\Train_Data\\Train_Keys.txt"
        VALID_DIR = r"D:\\machinelearning2\\ANL\\archive\\gnn\\Data\\Train_Data\\Valid_Keys.txt"
        TEST_DATA_DIR = r"D:\\machinelearning2\\ANL\\archive\\gnn\\Data\\Test_Data\\Test_Data.pkl"
        TEST_DIR = r"D:\\machinelearning2\\ANL\\archive\\gnn\\Data\\Test_Data\\Test_Keys.txt"
        PLOT_DIR = r"D:\\plots\\anl\\protonation\\"

    # data_dict = load_dict(name=DATA_DIR)
    # train, valid = load_data_split(TRAIN_DIR, VALID_DIR)
    # train = np.array(train)
    # valid = np.array(valid)
    # test_data_dict = load_dict(name=TEST_DATA_DIR)
    # test, _ = load_data_split(TEST_DIR, VALID_DIR)
    #
    # A_train, X_train, E_train, m_train, y_train = load_data_func(train, data_dict)
    # A_valid, X_valid, E_valid, m_valid, y_valid = load_data_func(valid, data_dict)
    #
    # with open("./protonation_data.pickle", "wb") as handle:
    #     pickle.dump([A_train, X_train, E_train, m_train, y_train], handle)
    #     pickle.dump([A_valid, X_valid, E_valid, m_valid, y_valid], handle)

    with open("./protonation_data.pickle", "rb") as handle:
        [A_train, X_train, E_train, m_train, y_train] = pickle.load(handle)
        [A_valid, X_valid, E_valid, m_valid, y_valid] = pickle.load(handle)
    print(f"================ Load Data ================")
    print(f"A_train shape: {np.shape(A_train)}")
    print(f"X_train shape: {np.shape(X_train)}")
    print(f"E_train shape: {np.shape(E_train)}")
    print(f"m_train shape: {np.shape(m_train)}")
    print(f"y_train shape: {np.shape(y_train)}")
    print(f"A_valid shape: {np.shape(A_valid)}")
    print(f"X_valid shape: {np.shape(X_valid)}")
    print(f"E_valid shape: {np.shape(E_valid)}")
    print(f"m_valid shape: {np.shape(m_valid)}")
    print(f"y_valid shape: {np.shape(y_valid)}")
    print(f"================ Load Data Finished ================")
    return ([X_train, A_train, E_train, m_train], y_train), ([X_valid, A_valid, E_valid, m_valid], y_valid)


if __name__ == '__main__':
    load_data()

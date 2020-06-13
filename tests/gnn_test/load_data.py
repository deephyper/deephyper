import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
import os
import numpy as np
# from Data_Processing import molecule
# from train_utils import load_data_split, load_dict, load_data_func
from spektral.datasets import citation
from spektral.utils import localpooling_filter
from sklearn.model_selection import train_test_split
from spektral.datasets import qm9
from spektral.utils import label_to_one_hot

# def load_data():
#     # define necessary parameters
#     elem_list = ['C', 'O', 'H', 'unknown']  # species of atoms
#     atom_fdim = len(elem_list) + 6 + 6 + 6 + 1  # number of atomic features
#     bond_fdim = 6  # number of bond features
#     max_nb = 6  # maximum number of neighbors
#     output_fdim = 1  # output feature dimension
#     max_atom = 31  # maximum atoms in the molecule
#
#     # define directories
#     if os.path.exists(r"/mnt/"):
#         DATA_DIR = r"/mnt/d/machinelearning2/ANL/gnn/Data/Train_Data/Total_Data.pkl"
#         TRAIN_DIR = r"/mnt/d/machinelearning2/ANL/gnn/Data/Train_Data/Train_Keys.txt"
#         VALID_DIR = r"/mnt/d/machinelearning2/ANL/gnn/Data/Train_Data/Valid_Keys.txt"
#         TEST_DATA_DIR = r"/mnt/d/machinelearning2/ANL/gnn/Data/Test_Data/Test_Data.pkl"
#         TEST_DIR = r"/mnt/d/machinelearning2/ANL/gnn/Data/Test_Data/Test_Keys.txt"
#     elif os.path.exists(r"D:\\machinelearning2\\"):
#         DATA_DIR = r"D:\\machinelearning2\\ANL\\gnn\\Data\\Train_Data\\Total_Data.pkl"
#         TRAIN_DIR = r"D:\\machinelearning2\\ANL\\gnn\\Data\\Train_Data\\Train_Keys.txt"
#         VALID_DIR = r"D:\\machinelearning2\\ANL\\gnn\\Data\\Train_Data\\Valid_Keys.txt"
#         TEST_DATA_DIR = r"D:\\machinelearning2\\ANL\\gnn\\Data\\Test_Data\\Test_Data.pkl"
#         TEST_DIR = r"D:\\machinelearning2\\ANL\\gnn\\Data\\Test_Data\\Test_Keys.txt"
#         PLOT_DIR = r"D:\\plots\\anl\\protonation\\"
#
#     data_dict = load_dict(name=DATA_DIR)
#     train, valid = load_data_split(TRAIN_DIR, VALID_DIR)
#     train = np.array(train)
#     valid = np.array(valid)
#     test_data_dict = load_dict(name=TEST_DATA_DIR)
#     test, _ = load_data_split(TEST_DIR, VALID_DIR)
#
#     A_train, X_train, E_train, m_train, y_train = load_data_func(train, data_dict)
#     A_valid, X_valid, E_valid, m_valid, y_valid = load_data_func(valid, data_dict)
#
#     print(f"================ Load Data ================")
#     print(f"A_train shape: {np.shape(A_train)}")
#     print(f"X_train shape: {np.shape(X_train)}")
#     print(f"E_train shape: {np.shape(E_train)}")
#     print(f"m_train shape: {np.shape(m_train)}")
#     print(f"y_train shape: {np.shape(y_train)}")
#     print(f"A_valid shape: {np.shape(A_valid)}")
#     print(f"X_valid shape: {np.shape(X_valid)}")
#     print(f"E_valid shape: {np.shape(E_valid)}")
#     print(f"m_valid shape: {np.shape(m_valid)}")
#     print(f"y_valid shape: {np.shape(y_valid)}")
#     print("\n")
#     return (A_train, X_train, E_train, m_train, y_train), (A_valid, X_valid, E_valid, m_valid, y_valid)

def load_data():
    # Load data
    A, X, E, y = qm9.load_data(return_type='numpy',
                               nf_keys='atomic_num',
                               ef_keys='type',
                               self_loops=True,
                               amount=1000)  # Set to None to train on whole dataset
    y = y[['cv']].values  # Heat capacity at 298.15K

    # Preprocessing
    uniq_X = np.unique(X)
    uniq_X = uniq_X[uniq_X != 0]
    X = label_to_one_hot(X, uniq_X)
    uniq_E = np.unique(E)
    uniq_E = uniq_E[uniq_E != 0]
    E = label_to_one_hot(E, uniq_E)

    # Parameters
    N = X.shape[-2]  # Number of nodes in the graphs
    F = X.shape[-1]  # Node features dimensionality
    S = E.shape[-1]  # Edge features dimensionality
    n_out = y.shape[-1]  # Dimensionality of the target
    learning_rate = 1e-3  # Learning rate for SGD
    epochs = 25  # Number of training epochs
    batch_size = 32  # Batch size
    es_patience = 5  # Patience fot early stopping

    # Train/test split
    A_train, A_test, \
    X_train, X_test, \
    E_train, E_test, \
    y_train, y_test = train_test_split(A, X, E, y, test_size=0.1)
    print(f"================ Load Data ================")
    print(f"A_train shape: {np.shape(A_train)}")
    print(f"X_train shape: {np.shape(X_train)}")
    print(f"E_train shape: {np.shape(E_train)}")
    print(f"y_train shape: {np.shape(y_train)}")
    print(f"A_test shape: {np.shape(A_test)}")
    print(f"X_test shape: {np.shape(X_test)}")
    print(f"E_test shape: {np.shape(E_test)}")
    print(f"y_test shape: {np.shape(y_test)}")
    return ((X_train, A_train, E_train), y_train), ((X_test, A_test, E_test), y_test)


if __name__ == "__main__":
    load_data()

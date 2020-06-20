import os
import numpy as np
import pickle
from protonation.gnnproton.Data_Processing import molecule
from protonation.gnnproton.train_utils import load_data_split, load_dict, load_data_func

# define necessary parameters
elem_list = ['C', 'O', 'H', 'unknown']  # species of atoms
atom_fdim = len(elem_list) + 6 + 6 + 6 + 1  # number of atomic features
bond_fdim = 6  # number of bond features
max_nb = 6  # maximum number of neighbors
output_fdim = 1  # output feature dimension
max_atom = 31  # maximum atoms in the molecule

# define directories
if os.path.exists(r"/mnt/d/machinelearning2"):
    DATA_DIR = r"/mnt/d/machinelearning2/ANL/deephyper/tests/protonation/protonation/gnnproton/"
else:
    DATA_DIR = r'/blues/gpfs/home/shengli.jiang/deephyper/tests/protonation/protonation/gnnproton/'

TEST_DATA_DIR = r'/mnt/d/machinelearning2/ANL/archive/gnn/Data/Test_Data/Test_Data.pkl'
TEST_DIR = r'/mnt/d/machinelearning2/ANL/archive/gnn/Data/Test_Data/Test_Keys.txt'
TRAIN_DIR = r'/mnt/d/machinelearning2/ANL/archive/gnn/Data/Train_Data/Train_Keys.txt'
test_data_dict = load_dict(name=TEST_DATA_DIR)
test, _ = load_data_split(TEST_DIR, TRAIN_DIR)
test = np.array(test)
A_test, X_test, E_test, m_test, y_test = load_data_func(test, test_data_dict)
#
# with open("./protonation_data_test.pickle", "wb") as handle:
#     pickle.dump([A_test, X_test, E_test, m_test, y_test], handle)
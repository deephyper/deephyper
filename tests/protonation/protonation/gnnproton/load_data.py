import os
import pickle


def load_data(edge=False):
    # define directories
    if os.path.exists(r"/mnt/d/machinelearning2"):
        DATA_DIR = r"/mnt/d/machinelearning2/ANL/deephyper/tests/protonation/protonation/gnnproton/"
    elif os.path.exists(r"/blues/gpfs/home/shengli.jiang"):
        DATA_DIR = r"/blues/gpfs/home/shengli.jiang/deephyper/tests/protonation/protonation/gnnproton/"
    else:
        DATA_DIR = r"D:\\machinelearning2\\ANL\\deephyper\\tests\\protonation\\protonation\\gnnproton\\"

    with open(DATA_DIR + "protonation_data.pickle", "rb") as handle:
        [A_train, X_train, E_train, m_train, y_train] = pickle.load(handle)
        [A_valid, X_valid, E_valid, m_valid, y_valid] = pickle.load(handle)
    A_train = A_train.reshape(A_train.shape[0], A_train.shape[1] * A_train.shape[2], 1)
    A_valid = A_valid.reshape(A_valid.shape[0], A_valid.shape[1] * A_valid.shape[2], 1)
    E_train = E_train.reshape(E_train.shape[0], E_train.shape[1] * E_train.shape[2], E_train.shape[3])
    E_valid = E_valid.reshape(E_valid.shape[0], E_valid.shape[1] * E_valid.shape[2], E_valid.shape[3])

    print(f"The task is protonation.")
    print(f"Max atom is {X_train.shape[1]}, N feature is {X_train.shape[2]}, E feature is {E_train.shape[-1]}.")
    print(f"Train size: {X_train.shape[0]}, valid size: {X_valid.shape[0]}, test size: None")

    if edge is False:
        return ([X_train, A_train, E_train], y_train), \
               ([X_valid, A_valid, E_valid], y_valid)
    else:
        return ([X_train, A_train, E_train], y_train), \
               ([X_valid, A_valid, E_valid], y_valid), \
               ([X_valid, A_valid, E_valid], y_valid), \
               'protonation', None


if __name__ == '__main__':
    load_data()
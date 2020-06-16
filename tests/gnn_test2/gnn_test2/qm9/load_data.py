from spektral.datasets import qm9
from spektral.utils import label_to_one_hot
import numpy as np
from sklearn.model_selection import train_test_split

np.random.seed(2018)


def load_data_qm9():
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
    return ([X_train, A_train, E_train], y_train), ([X_test, A_test, E_test], y_test)


if __name__ == '__main__':
    load_data_qm9()

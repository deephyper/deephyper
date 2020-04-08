import numpy as np


def load_data():
    from sklearn.datasets import load_breast_cancer

    X, y = load_breast_cancer(return_X_y=True)
    print(np.shape(X))
    print(np.shape(y))
    return X, y


if __name__ == "__main__":
    load_data()

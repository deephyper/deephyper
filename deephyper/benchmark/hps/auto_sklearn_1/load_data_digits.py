import numpy as np


def load_data():
    from sklearn.datasets import load_digits

    X, y = load_digits(return_X_y=True)
    print(np.shape(X))
    print(np.shape(y))
    return X, y


if __name__ == "__main__":
    load_data()

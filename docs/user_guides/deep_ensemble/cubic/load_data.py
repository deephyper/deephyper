import numpy as np
from sklearn.model_selection import train_test_split

def load_data_train_test(random_state=42):
    rs = np.random.RandomState(random_state)

    size_train, size_test = 40, 20
    x = rs.uniform(low=-4.0, high=4.0, size=size_train)
    eps = rs.normal(loc=0.0, scale=3.0, size=size_train)
    y = np.power(x, 3) + eps

    x_tst = np.linspace(-6.0, 6.0, size_test)
    y_tst = np.power(x_tst, 3)

    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)

    x_tst = x_tst.reshape(-1, 1)
    y_tst = y_tst.reshape(-1, 1)

    return (x, y), (x_tst, y_tst)


def load_data_train_valid(random_state=42):

    (x, y), _ = load_data_train_test(random_state=random_state)

    train_X, valid_X, train_y, valid_y = train_test_split(
        x, y, test_size=0.33, random_state=random_state
    )

    print(f'train_X shape: {np.shape(train_X)}')
    print(f'train_y shape: {np.shape(train_y)}')
    print(f'valid_X shape: {np.shape(valid_X)}')
    print(f'valid_y shape: {np.shape(valid_y)}')
    return (train_X, train_y), (valid_X, valid_y)

if __name__ == '__main__':
    load_data_train_valid()
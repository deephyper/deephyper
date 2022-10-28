from pprint import pformat
import numpy as np
import tensorflow as tf


def load_data(dim=10, size=100):
    """
    Generate data for linear function -sum(x_i).

    Return:
        Tuple of Numpy arrays: ``(train_X, train_y), (valid_X, valid_y)``.
    """
    rng = np.random.RandomState(42)
    size = 1000
    prop = 0.80
    a, b = 0, 100
    d = b - a
    x = np.array([a + rng.random(dim) * d for i in range(size)], dtype=np.float64)
    y = np.array([[np.sum(v)] for v in x], dtype=np.float64)

    sep_index = int(prop * size)

    sep_inputs = dim // 2  # we want two different inputs
    tX0, tX1 = x[:sep_index, :sep_inputs], x[:sep_index, sep_inputs:]
    vX0, vX1 = x[sep_index:, :sep_inputs], x[sep_index:, sep_inputs:]

    ty = y[:sep_index]
    vy = y[sep_index:]

    def train_gen():
        for x0, x1, y in zip(tX0, tX1, ty):
            yield ({"input_0": x0, "input_1": x1}, y)

    def valid_gen():
        for x0, x1, y in zip(vX0, vX1, vy):
            yield ({"input_0": x0, "input_1": x1}, y)

    res = {
        "train_gen": train_gen,
        "train_size": len(ty),
        "valid_gen": valid_gen,
        "valid_size": len(vy),
        "types": ({"input_0": tf.float64, "input_1": tf.float64}, tf.float64),
        "shapes": ({"input_0": (5,), "input_1": (5,)}, (1,)),
    }
    print("load_data:\n", pformat(res))
    return res


if __name__ == "__main__":
    load_data()

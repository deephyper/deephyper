import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop

import os
import sys

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
from load_data import load_data


def r2(y_true, y_pred):
    SS_res = tf.math.reduce_sum(tf.math.square(y_true - y_pred), axis=0)
    SS_tot = tf.math.reduce_sum(tf.math.square(y_true - tf.math.reduce_mean(y_true, axis=0)), axis=0)
    output_scores = 1 - SS_res / (SS_tot + tf.keras.backend.epsilon())
    r2 = tf.math.reduce_mean(output_scores)
    return r2


HISTORY = None


def run(point):
    global HISTORY
    (x_train, y_train), (x_valid, y_valid) = load_data()

    if point["activation"] == "identity":
        point["activation"] = None

    model = Sequential()
    model.add(
        Dense(
            point["units"],
            activation=point["activation"],
            input_shape=tuple(np.shape(x_train)[1:]),
        )
    )
    model.add(Dense(1))

    model.summary()

    model.compile(loss="mse", optimizer=RMSprop(lr=point["lr"]), metrics=[r2])

    history = model.fit(
        x_train,
        y_train,
        batch_size=64,
        epochs=1000,
        verbose=1,
        callbacks=[EarlyStopping(monitor="val_r2", mode="max", verbose=1, patience=10)],
        validation_data=(x_valid, y_valid),
    )

    HISTORY = history.history

    return history.history["val_r2"][-1]


if __name__ == "__main__":
    point = {"units": 10, "activation": "relu", "lr": 0.01}
    objective = run(point)
    print("objective: ", objective)
    import matplotlib.pyplot as plt

    plt.plot(HISTORY["val_r2"])
    plt.xlabel("Epochs")
    plt.ylabel("Objective: $R^2$")
    plt.grid()
    plt.show()

import numpy as np
import keras.backend as K
import keras
from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import RMSprop

import os
import sys

from polynome2.load_data import load_data

def r2(y_pred, y_true):
    SS_res = K.sum(K.square(y_true-y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res/(SS_tot + K.epsilon()))


HISTORY = None


def run(point):
    global HISTORY
    (x_train, y_train), (x_test, y_test) = load_data()

    model = Sequential()
    model.add(Dense(
        point['units'],
        activation=point['activation'],
        input_shape=tuple(np.shape(x_train)[1:])))
    model.add(Dense(1))

    model.summary()

    model.compile(loss='mse', optimizer=RMSprop(lr=point['lr']), metrics=[r2])

    history = model.fit(x_train, y_train,
                        batch_size=64,
                        epochs=1000,
                        verbose=1,
                        callbacks=[EarlyStopping(
                            monitor='val_r2',
                            mode='max',
                            verbose=1,
                            patience=10
                        )],
                        validation_data=(x_test, y_test))

    HISTORY = history.history

    return history.history['val_r2'][-1]


if __name__ == '__main__':
    from balsam.launcher.dag import current_job
    point = current_job.data['point']
    objective = run(point)
    current_job.data['dh_objective'] = objective
    current_job.save()
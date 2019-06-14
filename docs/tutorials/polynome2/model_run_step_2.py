import numpy as np
import keras.backend as K
import keras
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import RMSprop
from load_data import load_data
import os
import sys
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)


def r2(y_pred, y_true):
    SS_res = K.sum(K.square(y_true-y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res/(SS_tot + K.epsilon()))


def run(point):
    (x_train, y_train), (x_test, y_test) = load_data()

    model = Sequential()

    model.add(Dense(
        units=point['units_l1'],
        activation=point['activation_l1'],
        input_shape=tuple(np.shape(x_train)[1:])))
    model.add(Dropout(
        point['dropout_l1']))
    model.add(Dense(
        units=point['units_l2'],
        activation=point['activation_l2'],
        input_shape=tuple(np.shape(x_train)[1:])))
    model.add(Dropout(
        point['dropout_l2']))

    model.add(Dense(1))

    model.summary()

    model.compile(
        loss='mse',
        optimizer=RMSprop(
            lr=point['lr']
        ),
        metrics=[r2])

    history = model.fit(x_train, y_train,
                        batch_size=point['batch_size'],
                        epochs=1000,
                        verbose=1,
                        callbacks=[EarlyStopping(
                            monitor='val_r2',
                            mode='max',
                            verbose=1,
                            patience=10
                        )],
                        validation_data=(x_test, y_test))

    return max(history.history['val_r2'])


if __name__ == '__main__':
    res = run(dict(units_l1=10,
                   activation_l1='relu',
                   dropout_l1=0.1,
                   units_l2=10,
                   activation_l2='relu',
                   dropout_l2=0.1,
                   batch_size=64,
                   lr=0.01))
    print('res: ', res)

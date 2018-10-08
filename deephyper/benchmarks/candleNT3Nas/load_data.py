from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys
import shutil
import numpy as np, pandas as pd
from sklearn.preprocessing import MaxAbsScaler
from keras.utils import np_utils

from balsam.service.schedulers import JobEnv
from deephyper.benchmarks.candleNT3Nas import data_utils


HERE = os.path.dirname(os.path.abspath(__file__))

def load_data():
    dest = HERE+'/DATA'
    if JobEnv.host_type == 'THETA' and __name__ != '__main__':
        ram_path = '/dev/shm/data'
        if not os.path.isdir(ram_path):
            shutil.copytree(src=dest, dst=ram_path)
        dest = ram_path

    gParameters = {
        'data_url': 'ftp://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/normal-tumor/',
        'test_data': 'nt_test2.csv',
        'train_data': 'nt_train2.csv'
    }

    file_train = gParameters['train_data']
    file_test = gParameters['test_data']
    url = gParameters['data_url']

    train_path = data_utils.get_file(file_train, url + file_train, cache_subdir=dest)
    test_path = data_utils.get_file(file_test, url + file_test, cache_subdir=dest)

    print('Loading data...')
    df_train = (pd.read_csv(train_path, header=None).values).astype('float32')
    df_test = (pd.read_csv(test_path, header=None).values).astype('float32')
    print('done')

    print('df_train shape:', df_train.shape)
    print('df_test shape:', df_test.shape)

    seqlen = df_train.shape[1]

    df_y_train = df_train[:, 0].astype('int')
    df_y_test = df_test[:, 0].astype('int')

    Y_train = df_y_train
    Y_test = df_y_test

    df_x_train = df_train[:, 1:seqlen].astype(np.float32)
    df_x_test = df_test[:, 1:seqlen].astype(np.float32)

    X_train = df_x_train
    X_test = df_x_test

    scaler = MaxAbsScaler()
    mat = np.concatenate((X_train, X_test), axis=0)
    mat = scaler.fit_transform(mat)
    X_train = mat[:X_train.shape[0], :]
    X_test = mat[X_train.shape[0]:, :]
    X_train = np.reshape(X_train, (list(X_train.shape)))
    Y_train = np.reshape(Y_train, (list(Y_train.shape) + [1]))
    X_test = np.reshape(X_test, (list(X_test.shape)))
    Y_test = np.reshape(Y_test, (list(Y_test.shape) + [1]))

    print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

    return (X_train, Y_train), (X_test, Y_test)

if __name__ == '__main__':
    load_data()

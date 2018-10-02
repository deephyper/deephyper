"""
Created by Dipendra Jha (dipendra@u.northwestern.edu) on 7/16/18

Utilities for parsing PTB text files.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys
import numpy as np, pandas as pd
from sklearn.preprocessing import MaxAbsScaler
from keras.utils import np_utils
from deephyper.benchmarks.candleTC1Nas import data_utils

HERE = os.path.dirname(os.path.abspath(__file__))

def load_data(dest=None):
    if not dest: dest = HERE+'/DATA'

    gParameters = {'shuffle': True, 'verbose': False, 'conv': [128, 20, 1, 128, 10, 1], 'run_id': 'RUN.000', 'epochs': 400, 'alpha_dropout': False, 'out_act': 'softmax', 'logfile': None, 'dense': [200, 20], 'feature_subsample': 0, 'save': '.', 'data_url': 'ftp://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/type-class/', 'optimizer': 'sgd', 'timeout': -1, 'activation': 'relu', 'batch_size': 20, 'metrics': 'accuracy', 'gpus': [], 'train_bool': True, 'pool': [1, 10], 'loss': 'categorical_crossentropy', 'datatype': 'numpy.float32', 'drop': 0.1, 'config_file': '/Users/dipendra/Projects/Benchmarks/Pilot1/TC1/tc1_default_model.txt', 'test_data': 'type_18_300_test.csv', 'classes': 36, 'experiment_id': 'EXP.000', 'model_name': 'tc1', 'train_data': 'type_18_300_train.csv'}

    file_train = gParameters['train_data']
    file_test = gParameters['test_data']
    url = gParameters['data_url']
    train_path = data_utils.get_file(file_train, url + file_train, cache_subdir=dest)
    test_path = data_utils.get_file(file_test, url + file_test, cache_subdir=dest)

    print('Loading data...')
    print('train_path: {} test_path: {} gParameters: {}'.format(train_path, test_path, gParameters))
    if gParameters['feature_subsample'] > 0:
        usecols = list(range(gParameters['feature_subsample']))
    else:
        usecols = None
    df_train = (pd.read_csv(train_path, header=None, usecols=usecols).values).astype('float32')
    df_test = (pd.read_csv(test_path, header=None, usecols=usecols).values).astype('float32')
    print('done')

    print('df_train shape:', df_train.shape)
    print('df_test shape:', df_test.shape)

    seqlen = df_train.shape[1]

    df_y_train = df_train[:, 0].astype('int')
    df_y_test = df_test[:, 0].astype('int')

    #Y_train = np_utils.to_categorical(df_y_train, gParameters['classes'])
    #Y_test = np_utils.to_categorical(df_y_test, gParameters['classes'])

    Y_train = df_y_train
    Y_test = df_y_test

    df_x_train = df_train[:, 1:seqlen].astype(np.float32)
    df_x_test = df_test[:, 1:seqlen].astype(np.float32)

    #        X_train = df_x_train.as_matrix()
    #        X_test = df_x_test.as_matrix()

    X_train = df_x_train
    X_test = df_x_test

    scaler = MaxAbsScaler()
    mat = np.concatenate((X_train, X_test), axis=0)
    mat = scaler.fit_transform(mat)

    X_train = mat[:X_train.shape[0], :]
    X_test = mat[X_train.shape[0]:, :]
    X_train = np.reshape(X_train, (list(X_train.shape)+[1]))
    Y_train = np.reshape(Y_train, (list(Y_train.shape)))
    X_test = np.reshape(X_test, (list(X_test.shape)+[1]))
    Y_test = np.reshape(Y_test, (list(Y_test.shape)))

    print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

    return (X_train, Y_train), (X_test, Y_test)

if __name__ == '__main__':
    load_data()

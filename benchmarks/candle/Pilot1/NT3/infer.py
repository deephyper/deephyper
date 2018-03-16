from __future__ import print_function
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
import time
import gzip
import argparse
try:
    import configparser
except ImportError:
    import ConfigParser as configparser

from keras import backend as K

from keras.layers import Input, Dense, Dropout, Activation, Conv1D, MaxPooling1D, Flatten
from keras.optimizers import SGD, Adam, RMSprop
from keras.models import Sequential, Model, model_from_json, model_from_yaml
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler

TIMEOUT=3600 # in sec; set this to -1 for no timeoutA
 
# needed to find libs in ../common and ../../common
file_path = os.path.dirname(os.path.realpath(__file__))
lib_path = os.path.abspath(os.path.join(file_path, '..', 'common'))
sys.path.append(lib_path)
lib_path2 = os.path.abspath(os.path.join(file_path, '..', '..', 'common'))
sys.path.append(lib_path2)

import data_utils
import p1_common, p1_common_keras

from keras.models import Sequential, Model, model_from_json, model_from_yaml
from sklearn.metrics import accuracy_score

def load_data(train_path, test_path, gParameters):

    print('Loading data...')
    df_train = (pd.read_csv(train_path,header=None).values).astype('float32')
    df_test = (pd.read_csv(test_path,header=None).values).astype('float32')
    print('done')

    print('df_train shape:', df_train.shape)
    print('df_test shape:', df_test.shape)

    seqlen = df_train.shape[1]

    df_y_train = df_train[:,0].astype('int')
    df_y_test = df_test[:,0].astype('int')

    Y_train = np_utils.to_categorical(df_y_train,gParameters['classes'])
    Y_test = np_utils.to_categorical(df_y_test,gParameters['classes'])

    df_x_train = df_train[:, 1:seqlen].astype(np.float32)
    df_x_test = df_test[:, 1:seqlen].astype(np.float32)

    X_train = df_x_train
    X_test = df_x_test

    scaler = MaxAbsScaler()
    mat = np.concatenate((X_train, X_test), axis=0)
    mat = scaler.fit_transform(mat)

    X_train = mat[:X_train.shape[0], :]
    X_test = mat[X_train.shape[0]:, :]

    return X_train, Y_train, X_test, Y_test



output_dir = 'NT3'
model_name = 'nt3'
optimizer = 'sgd'
loss = 'categorical_crossentropy'
metrics = 'accuracy'
data_url = 'ftp://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/normal-tumor/'
train_data = 'nt_train2.csv'
test_data = 'nt_test2.csv'
classes=2

gParameters = {}
gParameters['loss'] = loss
gParameters['optimizer'] = optimizer
gParameters['metrics'] = metrics
gParameters['train_data'] = train_data
gParameters['test_data'] = test_data
gParameters['data_url'] = data_url
gParameters['classes'] = classes

# load json and create model
print (str(datetime.now()),  " loading model")
start = time.time()
json_file = open('{}/{}.model.json'.format(output_dir, model_name), 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model_json = model_from_json(loaded_model_json)

# load weights into new model
print (str(datetime.now()),  " loading weights")
loaded_model_json.load_weights('{}/{}.weights.h5'.format(output_dir, model_name))
print("Loaded json model from disk")
end = time.time()
print ('loading model elapsed time in seconds: ', end - start)

# load the test dataA
print (str(datetime.now()),  " loading data")
start = time.time()
file_train = gParameters['train_data']
file_test = gParameters['test_data']
url = gParameters['data_url']

train_file = data_utils.get_file(file_train, url+file_train, cache_subdir='Pilot1')
test_file = data_utils.get_file(file_test, url+file_test, cache_subdir='Pilot1')

X_train, Y_train, X_test, Y_test = load_data(train_file, test_file, gParameters)

# perform predictions on merged train and test data
X = np.concatenate((X_train, X_test), axis=0)
print ('X shape: ', X.shape)
print (str(datetime.now()),  " done loading data")
end = time.time()
print ('loading data elapsed time in seconds: ', end - start)

# this reshaping is critical for the Conv1D to work
X = np.expand_dims(X, axis=2)
print('X shape" ', X.shape)

# do prediction
print (str(datetime.now()),  " performing inference")
start = time.time()
loaded_model_json.compile(loss=gParameters['loss'],
    optimizer=gParameters['optimizer'],
    metrics=[gParameters['metrics']])
prediction = loaded_model_json.predict(X, verbose=0)
print (str(datetime.now()),  " done performing inferendce")
end = time.time()
print('prediction on ', X.shape[0], ' samples elapsed time in seconds: ', end - start)
print( prediction )

print (str(datetime.now()),  " done")

import sys
import os
import time

here = os.path.dirname(os.path.abspath(__file__))
top = os.path.dirname(os.path.dirname(os.path.dirname(here)))
sys.path.insert(top)

start = time.time()
import numpy as np
import pandas as pd
np.random.seed(10)

from numpy import concatenate
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM
from sklearn.pipeline import Pipeline
from keras.callbacks import Callback
from keras.callbacks import EarlyStopping
from keras import backend as K
from importlib import reload
from keras import optimizers
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, Dropout, concatenate
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD
from keras.layers import Input, Dense
from keras.models import Model

from keras import layers
from deephyper.benchmarks import keras_cmdline
load_time = time.time() - start
print(f"module import time: {load_time:.3f} seconds")


def run(param_dict):
    optimizer = keras_cmdline.return_optimizer(param_dict)
    print(param_dict)

    BATCH_SIZE = param_dict['batch_size']
    HIDDEN_SIZE = param_dict['hidden_size']
    NUNITS = param_dict['nunits']
    DROPOUT = param_dict['dropout']

    fpath = os.path.dirname(os.path.abspath(__file__))

    tag = 'ml-climate-hm-01'
    inp_df = pd.read_csv(fpath+'/data/1980-2005_2d_002_new.txt', sep="  ", header=None, engine='python')
    out_df1 = pd.read_csv(fpath+'/data/1980-2005_3d_vy_002.txt', sep=r"\s*", header=None, engine='python')
    out_df2 = pd.read_csv(fpath+'/data/1980-2005_3d_ux_002.txt', sep=r"\s*", header=None, engine='python')
    out_df3 = pd.read_csv(fpath+'/data/1980-2005_3d_wz_002.txt', sep=r"\s*", header=None, engine='python')
    out_df4 = pd.read_csv(fpath+'/data/1980-2005_3d_tk_002.txt', sep=r"\s*", header=None, engine='python')
    out_df5 = pd.read_csv(fpath+'/data/1980-2005_3d_qv_002.txt', sep=r"\s*", header=None, engine='python')
    out_df = pd.concat([out_df1, out_df2, out_df3, out_df4, out_df5], axis=1)
    inp_df = inp_df.iloc[0:72900,:]
    out_df = out_df.iloc[0:72900,:]

    print(inp_df.shape)
    print(out_df.shape)

    rlevels = 17 # we need 0 to 16 layers
    nlevels = out_df1.shape[1]
    nseries = int(out_df.shape[1]/nlevels)
    tlevels = out_df.shape[1]
    indices = range(tlevels)

    res = []
    for i in range(rlevels):
        for j in range(nseries):
            res.append(indices[i+(nlevels*j)])  
    print(res)
    print(len(res))
    req_out = res

    req_out = [int(x) for x in req_out]
    # load dataset
    dataset = out_df
    print(dataset.shape)
    values = dataset.values
    # specify columns to plot
    groups = req_out
    if len(req_out) > 60:
        groups = np.random.choice(groups, 60, replace=False)
        groups = np.sort(groups)    
    print(values.shape)

    def set_keras_backend(backend):
        if K.backend() != backend:
            os.environ['KERAS_BACKEND'] = backend
            reload(K)
            assert K.backend() == backend
    set_keras_backend("tensorflow")

    # load dataset
    out_df_sel = out_df.iloc[:,req_out]
    n_input = inp_df.shape[1]
    n_output = out_df_sel.shape[1]

    #dataset = out_df.iloc[:,1]
    dataset = pd.concat([inp_df, out_df_sel], axis=1)
    print(dataset.shape)
    values = dataset.values
    #print(values)
    # ensure all data is float
    values = values.astype('float32')
    # normalize features

    preprocessor = Pipeline([('stdscaler', StandardScaler()), ('minmax', MinMaxScaler(feature_range=(0, 1)))])
    reframed = preprocessor.fit_transform(values)
    print(reframed.shape)

    # split into train and test sets
    values = reframed
    n_train_hours = 40000
    n_val_hours = 10000
    train = values[:n_train_hours, :]
    validation = values[n_train_hours:(n_train_hours+n_val_hours),:]
    test = values[n_train_hours+n_val_hours:, :]
    # split into input and outputs
    train_X, train_y = train[:, range(n_input)], train[:, range(n_input, n_input+n_output)]
    validation_X, validation_y = validation[:, range(n_input)], validation[:, range(n_input, n_input+n_output)]
    test_X, test_y = test[:, range(n_input)], test[:, range(n_input, n_input+n_output)]
    print(train_X.shape, train_y.shape, validation_X.shape, validation_X.shape, test_X.shape, test_y.shape)

    # checkpoint
    filepath= '%s_weights.best.hdf5' % tag
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='auto')
    callbacks_list = [early_stopping]


    input_shape = train_X.shape[1:]
    inputs = Input(shape=input_shape)
    x = Dense(NUNITS, activation='relu')(inputs)
    x = Dropout(DROPOUT)(x)
    for j in range(HIDDEN_SIZE):
        x = Dense(NUNITS, activation='relu')(x)
        x = Dropout(DROPOUT)(x)
    x = Dense(NUNITS, activation='relu')(x)
    level_0 = Dense(nseries, name='level_0')(x)
    prev = level_0
    outputs_list = []
    outputs_list.append(prev) 
    for i in range(rlevels-1):
        merged = concatenate([inputs] + [prev])
        x = Dense(NUNITS, activation='relu')(merged)
        x = Dropout(DROPOUT)(x)
        for j in range(HIDDEN_SIZE):
            x = Dense(NUNITS, activation='relu')(x)
            x = Dropout(DROPOUT)(x)
        prev = Dense(nseries, name='level_%d' %(i+1))(x)
        outputs_list.append(prev)
    model = Model(inputs=inputs, outputs=concatenate(outputs_list))
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
    model.summary()

    history = model.fit(train_X, train_y, validation_data=(validation_X, validation_y), callbacks=callbacks_list, epochs=4, batch_size=64, verbose=1)

    yhat = model.predict(test_X)
    diff = test_y.ravel() - yhat.ravel()

    output = np.percentile(abs(diff), 95)
    print('OUTPUT: %f'% output)
    return output


def augment_parser(parser):
    parser.add_argument('--hidden_size', action='store', dest='hidden_size',
                        nargs='?', const=2, type=int, default='1',
                        help='number of hidden layers')
    parser.add_argument('--nunits', action='store', dest='nunits',
                        nargs='?', const=2, type=int, default='5',
                        help='number of units per hidden layer')
    return parser


if __name__ == "__main__":
    parser = keras_cmdline.create_parser()
    parser = augment_parser(parser)
    cmdline_args = parser.parse_args()
    param_dict = vars(cmdline_args)
    run(param_dict)

from __future__ import absolute_import

import numpy as np
import pandas as pd

import os
import sys
import gzip
import argparse

from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler

file_path = os.path.dirname(os.path.realpath(__file__))
lib_path = os.path.abspath(os.path.join(file_path, '..', '..', 'common'))
sys.path.append(lib_path)

from data_utils import get_file


# Seed for random generation -- default value
SEED = 7102
DEFAULT_DATATYPE = np.float32


def get_p1_file(link):
    fname = os.path.basename(link)
    return get_file(fname, origin=link, cache_subdir='Pilot1')


def get_default_neon_parse(parser):
    """Parse command-line arguments that are default in neon parser (and are common to all frameworks).
        Ignore if not present.

        Parameters
        ----------
        parser : python argparse
            parser for neon default command-line options
    """
    # Logging Level
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="increase output verbosity")
    parser.add_argument("-l", "--log", dest='logfile',
                        default=None,
                        help="log file")

    # Logging utilities
    parser.add_argument("-s", "--save_path", dest='save_path',
                        default=argparse.SUPPRESS, type=str,
                        help="file path to save model snapshots")

    # General behavior
    parser.add_argument("--model_file", dest='weight_path', type=str,
                        default=argparse.SUPPRESS,
                        help="specify trained model Pickle file")
    parser.add_argument("-d", "--data-type", dest='datatype',
                        default=argparse.SUPPRESS,
                        choices=['f16', 'f32', 'f64'],
                        help="default floating point")

    # Model definition
    # Model Architecture
    parser.add_argument("--dense", nargs='+', type=int,
                        default=argparse.SUPPRESS,
                        help="number of units in fully connected layers in an integer array")

    # Data preprocessing
    #parser.add_argument("--shuffle", action="store_true",
    #                    default=True,
    #                    help="randomly shuffle data set (produces different training and testing partitions each run depending on the seed)")

    # Training configuration
    parser.add_argument("-r", "--rng_seed", dest='rng_seed', type=int,
                        default=argparse.SUPPRESS,
                        help="random number generator seed")
    parser.add_argument("-e", "--epochs", type=int,
                        default=argparse.SUPPRESS,
                        help="number of training epochs")
    parser.add_argument("-z", "--batch_size", type=int,
                        default=argparse.SUPPRESS,
                        help="batch size")

    return parser


def get_p1_common_parser(parser):
    """Parse command-line arguments. Ignore if not present.

        Parameters
        ----------
        parser : python argparse
            parser for command-line options
    """

    # General behavior
    parser.add_argument("--train", dest='train_bool', action="store_true",
                        default=True, #type=bool,
                        help="train model")
    parser.add_argument("--evaluate", dest='eval_bool', action="store_true",
                        default=argparse.SUPPRESS, #type=bool,
                        help="evaluate model (use it for inference)")
    parser.add_argument("--timeout", type=int,
                        default=-1,
                        help="timeout in seconds")

    # Logging utilities
    parser.add_argument("--home_dir", dest='home_dir',
                        default=argparse.SUPPRESS, type=str,
                        help="set home directory")
    parser.add_argument("--save",
                        default=argparse.SUPPRESS, type=str,
                        help="prefix of output files")


    # Model definition
    # Model Architecture
    parser.add_argument("--conv", nargs='+', type=int,
                        default=argparse.SUPPRESS,
                        help="integer array describing convolution layers: conv1_filters, conv1_filter_len, conv1_stride, conv2_filters, conv2_filter_len, conv2_stride ...")
    parser.add_argument("--locally_connected", action="store_true",
                        default=argparse.SUPPRESS,
                        help="use locally connected layers instead of convolution layers")
    parser.add_argument("-a", "--activation",
                        default=argparse.SUPPRESS,
                        help="keras activation function to use in inner layers: relu, tanh, sigmoid...")

    # Processing between layers
    parser.add_argument("--drop", type=float,
                        default=argparse.SUPPRESS,
                        help="ratio of dropout used in fully connected layers")
    parser.add_argument("--pool", type=int,
                        default=argparse.SUPPRESS,
                        help="pooling layer length")
    parser.add_argument("--batch_normalization", action="store_true",
                        default=argparse.SUPPRESS,
                        help="use batch normalization")

    # Data preprocessing
    parser.add_argument("--scaling",
                        default=argparse.SUPPRESS,
                        choices=['minabs', 'minmax', 'std', 'none'],
                        help="type of feature scaling; 'minabs': to [-1,1]; 'minmax': to [0,1], 'std': standard unit normalization; 'none': no normalization")
    parser.add_argument("--shuffle", action="store_true",
                        default=True,
                        help="randomly shuffle data set (produces different training and testing partitions each run depending on the seed)")

    # Feature selection
    parser.add_argument("--feature_subsample", type=int,
                        default=argparse.SUPPRESS,
                        help="number of features to randomly sample from each category (cellline expression, drug descriptors, etc), 0 means using all features")

    # Training configuration
    parser.add_argument("--loss",
                        default=argparse.SUPPRESS,
                        help="keras loss function to use: mse, ...")
    parser.add_argument("--optimizer",
                        default=argparse.SUPPRESS,
                        help="keras optimizer to use: sgd, rmsprop, ...")
    parser.add_argument('--lr', dest='learning_rate', type=float,
                        default=argparse.SUPPRESS,
                        help='learning rate')
    parser.add_argument("--initialization",
                        default=argparse.SUPPRESS,
                        choices=['constant', 'uniform', 'normal', 'glorot_uniform', 'lecun_uniform', 'lecun_normal', 'he_normal'],
                        help="type of weight initialization; 'constant': to 0; 'uniform': to [-0.05,0.05], 'normal': mean 0, stddev 0.05; 'glorot_uniform': [-lim,lim] with lim = sqrt(6/(fan_in+fan_out)); 'lecun_uniform': [-lim,lim] with lim = sqrt(3/fan_in); 'lecun_normal': truncated normal with mean 0, stddev sqrt(1/fan_in); 'he_normal' : mean 0, stddev sqrt(2/fan_in)")
    parser.add_argument("--alpha_dropout", action='store_true',
                        help="use AlphaDropout instead of regular Dropout")
    parser.add_argument("--val_split", type=float,
                        default=argparse.SUPPRESS,
                        help="fraction of data to use in validation")
    parser.add_argument("--train_steps", type=int,
                        default=argparse.SUPPRESS,
                        help="overrides the number of training batches per epoch if set to nonzero")
    parser.add_argument("--val_steps", type=int,
                        default=argparse.SUPPRESS,
                        help="overrides the number of validation batches per epoch if set to nonzero")
    parser.add_argument("--test_steps", type=int,
                        default=argparse.SUPPRESS,
                        help="overrides the number of test batches per epoch if set to nonzero")
    parser.add_argument("--train_samples", action="store",
                        default=argparse.SUPPRESS, type=int,
                        help="overrides the number of training samples if set to nonzero")
    parser.add_argument("--val_samples", action="store",
                        default=argparse.SUPPRESS, type=int,
                        help="overrides the number of validation samples if set to nonzero")


    # Backend configuration
    parser.add_argument("--gpus", action="store", nargs='*',
                        default=[], type=int,
                        help="set IDs of GPUs to use")

    # solr monitor configuration
    parser.add_argument("--experiment_id", type=str,
                        default="EXP.000",
                        help="experiment id")

    parser.add_argument("--run_id", type=str,
                        default="RUN.000",
                        help="run id")


    return parser



def args_overwrite_config(args, config):
    """Overwrite configuration parameters with
        parameters specified via command-line.

        Parameters
        ----------
        args : python argparse
            parameters specified via command-line
        config : python dictionary
            parameters read from configuration file
    """

    params = config

    args_dict = vars(args)

    for key in args_dict.keys():
        params[key] = args_dict[key]

    if 'datatype' not in params:
        params['datatype'] = DEFAULT_DATATYPE
    else:
        if params['datatype'] in set(['f16', 'f32', 'f64']):
            params['datatype'] = get_choice(params['datatype'])

    return params



def keras_default_config():
    """Defines parameters that intervine in different functions using the keras defaults.
        This helps to keep consistency in parameters between frameworks.
    """

    kerasDefaults = {}

    # Optimizers
    #kerasDefaults['clipnorm']=?            # Maximum norm to clip all parameter gradients
    #kerasDefaults['clipvalue']=?          # Maximum (minimum=-max) value to clip all parameter gradients
    kerasDefaults['decay_lr']=0.            # Learning rate decay over each update
    kerasDefaults['epsilon']=1e-8           # Factor to avoid divide by zero (fuzz factor)
    kerasDefaults['rho']=0.9                # Decay parameter in some optmizer updates (rmsprop, adadelta)
    kerasDefaults['momentum_sgd']=0.        # Momentum for parameter update in sgd optimizer
    kerasDefaults['nesterov_sgd']=False     # Whether to apply Nesterov momentum in sgd optimizer
    kerasDefaults['beta_1']=0.9             # Parameter in some optmizer updates (adam, adamax, nadam)
    kerasDefaults['beta_2']=0.999           # Parameter in some optmizer updates (adam, adamax, nadam)
    kerasDefaults['decay_schedule_lr']=0.004# Parameter for nadam optmizer

    # Initializers
    kerasDefaults['minval_uniform']=-0.05   #  Lower bound of the range of random values to generate
    kerasDefaults['maxval_uniform']=0.05    #  Upper bound of the range of random values to generate
    kerasDefaults['mean_normal']=0.         #  Mean of the random values to generate
    kerasDefaults['stddev_normal']=0.05     #  Standard deviation of the random values to generate


    return kerasDefaults


def get_choice(name):
    """ Maps name string to the right type of argument
    """
    mapping = {}

    # dtype
    mapping['f16'] = np.float16
    mapping['f32'] = np.float32
    mapping['f64'] = np.float64

    mapped = mapping.get(name)
    if not mapped:
        raise Exception('No mapping found for "{}"'.format(name))

    return mapped


def scale_array(mat, scaling=None):
    """Scale data included in numpy array.

        Parameters
        ----------
        mat : numpy array
            array to scale
        scaling : 'maxabs', 'minmax', 'std', or None, optional (default 'None')
            type of scaling to apply
    """

    if scaling is None or scaling.lower() == 'none':
        return mat

    # Scaling data
    if scaling == 'maxabs':
        # Normalizing -1 to 1
        scaler = MaxAbsScaler(copy=False)
    elif scaling == 'minmax':
        # Scaling to [0,1]
        scaler = MinMaxScaler(copy=False)
    else:
        # Standard normalization
        scaler = StandardScaler(copy=False)

    return scaler.fit_transform(mat)


def impute_and_scale_array(mat, scaling=None):
    """Impute missing values with mean and scale data included in numpy array.

        Parameters
        ----------
        mat : numpy array
            array to scale
        scaling : 'maxabs', 'minmax', 'std', or None, optional (default 'None')
            type of scaling to apply
    """

    imputer = Imputer(strategy='mean', axis=0, copy=False)
    imputer.fit_transform(mat)
    #mat = imputer.fit_transform(mat)

    return scale_array(mat, scaling)


def load_X_data(path, train_filename, test_filename,
                shuffle=False, scaling=None,
                drop_cols=None, usecols=None, n_cols=None, onehot_cols=None,
                validation_split=None, dtype=DEFAULT_DATATYPE, seed=SEED):

    train_path = get_p1_file(path + train_filename)
    test_path = get_p1_file(path + test_filename)

    # compensates for the columns to drop if there is a feature subselection
    if usecols is None:
        usecols = list(range(n_cols + len(drop_cols))) if n_cols else None

    df_train = pd.read_csv(train_path, engine='c', usecols=usecols)
    df_test = pd.read_csv(test_path, engine='c', usecols=usecols)

    # Drop specified columns
    if drop_cols is not None:
        for col in drop_cols:
            df_train.drop(col, axis=1, inplace=True)
            df_test.drop(col, axis=1, inplace=True)

    if onehot_cols is not None:
        for col in onehot_cols:
            df_cat = pd.concat([df_train[col], df_test[col]])
            df_dummy = pd.get_dummies(df_cat, prefix=col, prefix_sep=': ')
            df_train = pd.concat([df_dummy[:df_train.shape[0]], df_train.drop(col, axis=1)], axis=1)
            df_test = pd.concat([df_dummy[df_train.shape[0]:], df_test.drop(col, axis=1)], axis=1)

    if shuffle:
        df_train = df_train.sample(frac=1, random_state=seed)
        df_test = df_test.sample(frac=1, random_state=seed)

    X_train = df_train.values.astype(dtype)
    X_test = df_test.values.astype(dtype)

    mat = np.concatenate((X_train, X_test), axis=0)
    # Scale data
    if scaling is not None:
        mat = scale_array(mat, scaling)

    n = X_train.shape[0]
    X_train = mat[:n, :]
    X_test = mat[n:, :]

    if validation_split and validation_split > 0 and validation_split < 1:
        n_val = int(n * validation_split)
        X_val = X_train[-n_val:, :]
        X_train = X_train[:-n_val, :]
        return X_train, X_val, X_test
    else:
        return X_train, X_test


def test_load_csv_data():
    train_path = '_train_load_data.csv'
    test_path = '_test_load_data.csv'

    nrows = 5
    df1 = pd.DataFrame({'y': np.random.randint(0, 9, nrows),
                        'x0': np.random.randint(30, 39, nrows),
                        'x1': np.random.randint(-10, 10, nrows) / 10.,
                        'x2': np.random.randint(0, 10, nrows),
                        'x3': np.random.randint(0, 10, nrows),
                        'x4': np.random.randint(0, 10, nrows),
                        'x5': np.random.randint(0, 100, nrows) / 10.,
                        'x6': np.random.randint(-10, 10, nrows),
                        'z1': pd.Categorical(['cat'] * 2 + ['dog'] * (nrows - 2)),
                        'z2': np.random.randint(0, 3, nrows)})

    nrows = 2
    df2 = pd.DataFrame({'y': np.random.randint(0, 9, nrows),
                        'x0': np.random.randint(30, 39, nrows),
                        'x1': np.random.randint(-10, 10, nrows) / 10.,
                        'x2': np.random.randint(0, 10, nrows),
                        'x3': np.random.randint(0, 3, nrows),
                        'x4': np.random.randint(0, 10, nrows),
                        'x5': np.random.randint(0, 100, nrows) / 10.,
                        'x6': np.random.randint(-10, 10, nrows),
                        'z1': pd.Categorical(['cat'] * 2 + ['dog'] * (nrows - 2)),
                        'z2': np.random.randint(0, 3, nrows)})

    df1.to_csv(train_path, index=False)
    df2.to_csv(test_path, index=False)

    x_train = load_csv_data(train_path)

    x_train, x_test = load_csv_data(train_path, test_path)

    x_train, x_val, x_test = load_csv_data(train_path, test_path, validation_split=0.2)

    x_train, x_val, x_test = load_csv_data(train_path, test_path, validation_split=0.2, shuffle=True)

    x_train, y_train = load_csv_data(train_path, y_cols=['y'])

    x_train, y_train = load_csv_data(train_path, y_cols=[7])

    x_train, y_train = load_csv_data(train_path, y_cols=[7, 8, 9])

    x_train, y_train = load_csv_data(train_path, y_cols=['y', 'z1'])

    x_train, y_train = load_csv_data(train_path, y_cols=['y', 'z1'], onehot_cols=['z1'])

    x_train, y_train = load_csv_data(train_path, y_cols=['y'], onehot_cols=['z1'])

    x_train, y_train = load_csv_data(train_path, y_cols=['y'], drop_cols=['x5', 'z1'])

    x_train, y_train = load_csv_data(train_path, y_cols=['y'], drop_cols=['x5', 'z1'], return_dataframe=False)

    x_train, y_train = load_csv_data(train_path, y_cols=['y'], x_cols=['x3', 'x5'])

    x_train, y_train = load_csv_data(train_path, y_cols=['y'], x_cols=['x3', 'x6'], n_cols=3)

    x_train, y_train = load_csv_data(train_path, y_cols=['y'], x_cols=['x3', 'x6'], onehot_cols=['z1'], n_cols=3)

    x_train, y_train = load_csv_data(train_path, y_cols=['y'], x_cols=['x3', 'x6'], onehot_cols=['z1'], n_cols=4, random_cols=True)

    x_train, y_train = load_csv_data(train_path, y_cols=['y'], onehot_cols=['z1'], n_cols=3)

    x_train, y_train = load_csv_data(train_path, y_cols=['y'], onehot_cols=['z1'], n_cols=3, random_cols=True)

    x_train, y_train = load_csv_data(train_path, y_cols=['y'], onehot_cols=['z1', 'x3'], n_cols=2)

    x_train, y_train = load_csv_data(train_path, y_cols=['y'], onehot_cols=['z1', 'x3'], n_cols=1)

    x_train, y_train = load_csv_data(train_path, y_cols=['y'], onehot_cols=['z1', 'x3'], n_cols=1, random_cols=True)

    x_train, y_train = load_csv_data(train_path, y_cols=['y'], onehot_cols=['z1', 'x3'], n_cols=1, random_cols=True)

    x_train, y_train = load_csv_data(train_path, y_cols=['y'], onehot_cols=['z1', 'y'])

    x_train, y_train = load_csv_data(train_path, y_cols=['y'], onehot_cols=['x3', 'z1', 'y'])

    x_train, y_train = load_csv_data(train_path, y_cols=['y', 'z1', 'z2'], onehot_cols=['x3', 'z1', 'y'])

    x_train, y_train, x_test, y_test = load_csv_data(train_path, test_path, y_cols=['y', 'z1', 'z2'], onehot_cols=['x3', 'z1', 'y'])

    x_train, y_train, x_val, y_val, x_test, y_test = load_csv_data(train_path, test_path, y_cols=['y', 'z1', 'z2'], onehot_cols=['x3', 'z1', 'y'], validation_split=0.4)

    x_train, y_train, x_val, y_val, x_test, y_test = load_csv_data(train_path, test_path, y_cols=['y', 'z1', 'z2'], onehot_cols=['x3', 'z1', 'y'], validation_split=0.4, return_dataframe=False)

    x_train, y_train, x_val, y_val, x_test, y_test = load_csv_data(train_path, test_path, scaling='minmax', y_cols=['y', 'z1', 'z2'], onehot_cols=['x3', 'z1', 'y'], validation_split=0.4)

    # train_path = '/home/fangfang/Benchmarks/Pilot1/P1B1/tmp.csv'
    # test_path = None
    # y_cols = [0]
    # y_cols = ['y']
    # x_cols = None
    # drop_cols = [3, 9]
    # drop_cols = None
    # n_cols = 3
    # n_cols = None
    # random_cols = True
    # random_cols = False
    # onehot_cols = None
    # # onehot_cols = None
    # dtype = None
    # shuffle = False
    # shuffle = True
    # scaling = None
    # scaling = 'minmax'
    # return_dataframe = True
    # validation_split = None
    # sep = ','


def load_csv_data(train_path, test_path=None, sep=',', nrows=None,
                  x_cols=None, y_cols=None, drop_cols=None,
                  onehot_cols=None, n_cols=None, random_cols=False,
                  shuffle=False, scaling=None, dtype=None,
                  validation_split=None, return_dataframe=True,
                  return_header=False, seed=2017):

    """Load training and test data from CSV

        Parameters
        ----------
        train_path : path
            training file path
    """

    if x_cols is None and drop_cols is None and n_cols is None:
        usecols = None
        y_names = None
    else:
        df_cols = pd.read_csv(train_path, engine='c', sep=sep, nrows=0)
        df_x_cols = df_cols.copy()
        # drop columns by name or index
        if y_cols is not None:
            df_x_cols = df_x_cols.drop(df_cols[y_cols], axis=1)
        if drop_cols is not None:
            df_x_cols = df_x_cols.drop(df_cols[drop_cols], axis=1)

        reserved = []
        if onehot_cols is not None:
            reserved += onehot_cols
        if x_cols is not None:
            reserved += x_cols

        nx = df_x_cols.shape[1]
        if n_cols and n_cols < nx:
            if random_cols:
                indexes = sorted(np.random.choice(list(range(nx)), n_cols, replace=False))
            else:
                indexes = list(range(n_cols))
            x_names = list(df_x_cols[indexes])
            unreserved = [x for x in x_names if x not in reserved]
            n_keep = np.maximum(n_cols - len(reserved), 0)
            combined = reserved + unreserved[:n_keep]
            x_names = [x for x in df_x_cols if x in combined]
        elif x_cols is not None:
            x_names = list(df_x_cols[x_cols])
        else:
            x_names = list(df_x_cols.columns)

        usecols = x_names
        if y_cols is not None:
            y_names = list(df_cols[y_cols])
            usecols = y_names + x_names

    df_train = pd.read_csv(train_path, engine='c', sep=sep, nrows=nrows, usecols=usecols)
    if test_path:
        df_test = pd.read_csv(test_path, engine='c', sep=sep, nrows=nrows, usecols=usecols)
    else:
        df_test = df_train[0:0].copy()

    if y_cols is None:
        y_names = []
    elif y_names is None:
        y_names = list(df_train[0:0][y_cols])

    if shuffle:
        df_train = df_train.sample(frac=1, random_state=seed)
        if test_path:
            df_test = df_test.sample(frac=1, random_state=seed)

    df_cat = pd.concat([df_train, df_test])
    df_y = df_cat[y_names]
    df_x = df_cat.drop(y_names, axis=1)

    if onehot_cols is not None:
        for col in onehot_cols:
            if col in y_names:
                df_dummy = pd.get_dummies(df_y[col], prefix=col, prefix_sep=':')
                df_y = pd.concat([df_dummy, df_y.drop(col, axis=1)], axis=1)
                # print(df_dummy.columns)
            else:
                df_dummy = pd.get_dummies(df_x[col], prefix=col, prefix_sep=':')
                df_x = pd.concat([df_dummy, df_x.drop(col, axis=1)], axis=1)

    if scaling is not None:
        mat = scale_array(df_x.values, scaling)
        df_x = pd.DataFrame(mat, index=df_x.index, columns=df_x.columns, dtype=dtype)

    n_train = df_train.shape[0]

    x_train = df_x[:n_train]
    y_train = df_y[:n_train]
    x_test = df_x[n_train:]
    y_test = df_y[n_train:]

    return_y = y_cols is not None
    return_val = validation_split and validation_split > 0 and validation_split < 1
    return_test = test_path

    if return_val:
        n_val = int(n_train * validation_split)
        x_val = x_train[-n_val:]
        y_val = y_train[-n_val:]
        x_train = x_train[:-n_val]
        y_train = y_train[:-n_val]

    ret = [x_train]
    ret = ret + [y_train] if return_y else ret
    ret = ret + [x_val] if return_val else ret
    ret = ret + [y_val] if return_y and return_val else ret
    ret = ret + [x_test] if return_test else ret
    ret = ret + [y_test] if return_y and return_test else ret

    if not return_dataframe:
        ret = [x.values for x in ret]

    if return_header:
        ret = ret + [df_x.columns.tolist(), df_y.columns.tolist()]

    return tuple(ret) if len(ret) > 1 else ret


def load_Xy_data(path, train_filename, test_filename, y_col=0,
                 shuffle=False, scaling=None, onehot_y=False, random_cols=False,
                 drop_cols=None, usecols=None, n_cols=None, onehot_cols=None,
                 validation_split=None, dtype=DEFAULT_DATATYPE, seed=SEED):

    train_path = get_p1_file(path + train_filename)
    test_path = get_p1_file(path + test_filename)

    X_train = df_train.values.astype(dtype)
    X_test = df_test.values.astype(dtype)

    mat = np.concatenate((X_train, X_test), axis=0)
    # Scale data
    if scaling is not None:
        mat = scale_array(mat, scaling)

    n = X_train.shape[0]
    X_train = mat[:n, :]
    X_test = mat[n:, :]

    if validation_split and validation_split > 0 and validation_split < 1:
        n_val = int(n * validation_split)
        X_val = X_train[-n_val:, :]
        X_train = X_train[:-n_val, :]
        return X_train, X_val, X_test
    else:
        return X_train, X_test


def load_Xy_one_hot_data(path, train_filename, test_filename,
                        class_col=None, drop_cols=None, n_cols=None, shuffle=False, scaling=None,
                        dtype=DEFAULT_DATATYPE, seed=SEED):

    assert class_col != None

    train_path = get_p1_file(path + train_filename)
    test_path = get_p1_file(path + test_filename)

    # compensates for the columns to drop if there is a feature subselection
    usecols = list(range(n_cols + len(drop_cols))) if n_cols else None

    df_train = pd.read_csv(train_path, engine='c', usecols=usecols)
    df_test = pd.read_csv(test_path, engine='c', usecols=usecols)

    if shuffle:
        df_train = df_train.sample(frac=1, random_state=seed)
        df_test = df_test.sample(frac=1, random_state=seed)

    # Get class
    y_train = pd.get_dummies(df_train[class_col]).values
    y_test = pd.get_dummies(df_test[class_col]).values

    # Drop specified columns
    if drop_cols is not None:
        for col in drop_cols:
            df_train.drop(col, axis=1, inplace=True)
            df_test.drop(col, axis=1, inplace=True)

    # Convert from pandas dataframe to numpy array
    X_train = df_train.values.astype(dtype)
    print("X_train dtype: ", X_train.dtype)
    X_test = df_test.values.astype(dtype)
    print("X_test dtype: ", X_test.dtype)
    # Concatenate training and testing to scale data
    mat = np.concatenate((X_train, X_test), axis=0)
    print("mat dtype: ", mat.dtype)
    # Scale data
    if scaling is not None:
        mat = scale_array(mat, scaling)
    # Recover training and testing splits after scaling
    X_train = mat[:X_train.shape[0], :]
    X_test = mat[X_train.shape[0]:, :]

    return (X_train, y_train), (X_test, y_test)


def load_Xy_one_hot_data2(path, train_filename, test_filename,
                    class_col=None, drop_cols=None, n_cols=None, shuffle=False, scaling=None,
                    validation_split=0.1, dtype=DEFAULT_DATATYPE, seed=SEED):

    assert class_col != None

    train_path = get_p1_file(path + train_filename)
    test_path = get_p1_file(path + test_filename)

    # compensates for the columns to drop if there is a feature subselection
    usecols = list(range(n_cols + len(drop_cols))) if n_cols else None

    df_train = pd.read_csv(train_path, engine='c', usecols=usecols)
    df_test = pd.read_csv(test_path, engine='c', usecols=usecols)

    if shuffle:
        df_train = df_train.sample(frac=1, random_state=seed)
        df_test = df_test.sample(frac=1, random_state=seed)

    # Get class
    y_train = pd.get_dummies(df_train[class_col]).values
    y_test = pd.get_dummies(df_test[class_col]).values

    # Drop specified columns
    if drop_cols is not None:
        for col in drop_cols:
            df_train.drop(col, axis=1, inplace=True)
            df_test.drop(col, axis=1, inplace=True)

    # Convert from pandas dataframe to numpy array
    X_train = df_train.values.astype(dtype)
    X_test = df_test.values.astype(dtype)
    # Concatenate training and testing to scale data
    mat = np.concatenate((X_train, X_test), axis=0)
    # Scale data
    if scaling is not None:
        mat = scale_array(mat, scaling)
    # Separate training in training and validation splits after scaling
    sizeTrain = X_train.shape[0]
    X_test = mat[sizeTrain:, :]
    numVal = int(sizeTrain * validation_split)
    X_val = mat[:numVal, :]
    X_train = mat[numVal:sizeTrain, :]
    # Analogously separate y in training in training and validation splits
    y_val = y_train[:numVal, :]
    y_train = y_train[numVal:sizeTrain, :]

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)



def load_Xy_data2(path, train_filename, test_filename, class_col=None, drop_cols=None, n_cols=None, shuffle=False, scaling=None,
                  validation_split=0.1, dtype=DEFAULT_DATATYPE, seed=SEED):

    assert class_col != None

    (X_train, y_train_oh), (X_val, y_val_oh), (X_test, y_test_oh) = load_Xy_one_hot_data2(path, train_filename, test_filename,
                                                                                 class_col, drop_cols, n_cols, shuffle, scaling,
                                                                                 validation_split, dtype, seed)

    y_train = convert_to_class(y_train_oh)
    y_val = convert_to_class(y_val_oh)
    y_test = convert_to_class(y_test_oh)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)



def convert_to_class(y_one_hot, dtype=int):
    maxi = lambda a: a.argmax()
    iter_to_na = lambda i: np.fromiter(i, dtype=dtype)
    return np.array([maxi(a) for a in y_one_hot])

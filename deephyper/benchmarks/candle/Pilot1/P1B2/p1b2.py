from __future__ import print_function

import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score

import os
import sys
import logging
import argparse
try:
    import configparser
except ImportError:
    import ConfigParser as configparser


file_path = os.path.dirname(os.path.realpath(__file__))
lib_path = os.path.abspath(os.path.join(file_path, '..', 'common'))
sys.path.append(lib_path)
lib_path2 = os.path.abspath(os.path.join(file_path, '..', '..', 'common'))
sys.path.append(lib_path2)

import p1_common

url_p1b2 = 'http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/P1B2/'
file_train = 'P1B2.train.csv'
file_test = 'P1B2.test.csv'

logger = logging.getLogger(__name__)


def common_parser(parser):

    parser.add_argument("--config_file", dest='config_file', type=str,
                        default=os.path.join(file_path, 'p1b2_default_model.txt'),
                        help="specify model configuration file")

    # Parse has been split between arguments that are common with the default neon parser
    # and all the other options
    parser = p1_common.get_default_neon_parse(parser)
    parser = p1_common.get_p1_common_parser(parser)

    return parser


def read_config_file(file):
    config=configparser.ConfigParser()
    config.read(file)
    section=config.sections()
    fileParams={}

    fileParams['activation']=eval(config.get(section[0],'activation'))
    fileParams['batch_size']=eval(config.get(section[0],'batch_size'))
    fileParams['dense']=eval(config.get(section[0],'dense'))
    fileParams['drop']=eval(config.get(section[0],'drop'))
    fileParams['epochs']=eval(config.get(section[0],'epochs'))
    fileParams['feature_subsample']=eval(config.get(section[0],'feature_subsample'))
    fileParams['initialization']=eval(config.get(section[0],'initialization'))
    fileParams['learning_rate']=eval(config.get(section[0], 'learning_rate'))
    fileParams['loss']=eval(config.get(section[0],'loss'))
    fileParams['optimizer']=eval(config.get(section[0],'optimizer'))
    fileParams['penalty']=eval(config.get(section[0],'penalty'))
    fileParams['rng_seed']=eval(config.get(section[0],'rng_seed'))
    fileParams['scaling']=eval(config.get(section[0],'scaling'))
    fileParams['validation_split']=eval(config.get(section[0],'validation_split'))

    # parse the remaining values
    for k, v in config.items(section[0]):
        if not k in fileParams:
            fileParams[k] = eval(v)

    return fileParams


def extension_from_parameters(params, framework):
    """Construct string for saving model with annotation of parameters"""
    ext = framework
    ext += '.A={}'.format(params['activation'])
    ext += '.B={}'.format(params['batch_size'])
    ext += '.D={}'.format(params['drop'])
    ext += '.E={}'.format(params['epochs'])
    if params['feature_subsample']:
        ext += '.F={}'.format(params['feature_subsample'])
    for i, n in enumerate(params['dense']):
        if n:
            ext += '.D{}={}'.format(i+1, n)
    ext += '.S={}'.format(params['scaling'])

    return ext


def load_data_one_hot(params, seed):
    return p1_common.load_Xy_one_hot_data2(url_p1b2, file_train, file_test, class_col=['cancer_type'],
                                           drop_cols=['case_id', 'cancer_type'],
                                           n_cols=params['feature_subsample'],
                                           shuffle=params['shuffle'],
                                           scaling=params['scaling'],
                                           validation_split=params['validation_split'],
                                           dtype=params['datatype'],
                                           seed=seed)


def load_data(params, seed):
    return p1_common.load_Xy_data2(url_p1b2, file_train, file_test, class_col=['cancer_type'],
                                  drop_cols=['case_id', 'cancer_type'],
                                  n_cols=params['feature_subsample'],
                                  shuffle=params['shuffle'],
                                  scaling=params['scaling'],
                                  validation_split=params['validation_split'],
                                  dtype=params['datatype'],
                                  seed=seed)


def evaluate_accuracy_one_hot(y_pred, y_test):
    def map_max_indices(nparray):
        maxi = lambda a: a.argmax()
        iter_to_na = lambda i: np.fromiter(i, dtype=np.float)
        return np.array([maxi(a) for a in nparray])
    ya, ypa = tuple(map(map_max_indices, (y_test, y_pred)))
    accuracy = accuracy_score(ya, ypa)
    # print('Accuracy: {}%'.format(100 * accuracy))
    return {'accuracy': accuracy}


def evaluate_accuracy(y_pred, y_test):
    accuracy = accuracy_score(y_test, y_pred)
    # print('Accuracy: {}%'.format(100 * accuracy))
    return {'accuracy': accuracy}

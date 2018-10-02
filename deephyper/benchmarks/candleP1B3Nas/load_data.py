#! /usr/bin/env python

"""Multilayer Perceptron for drug response problem"""

from __future__ import division, print_function

import argparse
import csv
import logging
import sys

import numpy as np


from deephyper.benchmarks.candleP1B3Nas import p1b3
from deephyper.benchmarks.candleP1B3Nas import p1_common
from deephyper.benchmarks.candleP1B3Nas import p1_common_keras

def get_p1b3_parser():
    parser = argparse.ArgumentParser(prog='p1b3_baseline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Train Drug Response Regressor - Pilot 1 Benchmark 3')

    return p1b3.common_parser(parser)

def initialize_parameters():
    # Get command-line parameters
    parser = get_p1b3_parser()
    args = parser.parse_args()
    #print('Args:', args)
    # Get parameters from configuration file
    fileParameters = p1b3.read_config_file(args.config_file)
    #print ('Params:', fileParameters)
    # Consolidate parameter set. Command-line parameters overwrite file configuration
    gParameters = p1_common.args_overwrite_config(args, fileParameters)
    return gParameters

def load_data(batch_size):
    """
    Runs the model using the specified set of parameters

    Args:
       gParameters: a python dictionary containing the parameters (e.g. epoch)
       to run the model with.
    """
    gParameters = initialize_parameters()
    seed = gParameters['rng_seed']
    gParameters['batch_size'] = batch_size
    print(f'batch_size: {gParameters["batch_size"]}')

    # Build dataset loader object
    loader = p1b3.DataLoader(seed=seed, dtype=gParameters['datatype'],
                             val_split=gParameters['validation_split'],
                             test_cell_split=gParameters['test_cell_split'],
                             cell_features=gParameters['cell_features'],
                             drug_features=gParameters['drug_features'],
                             feature_subsample=gParameters['feature_subsample'],
                             scaling=gParameters['scaling'],
                             scramble=gParameters['scramble'],
                             min_logconc=gParameters['min_logconc'],
                             max_logconc=gParameters['max_logconc'],
                             subsample=gParameters['subsample'],
                             category_cutoffs=gParameters['category_cutoffs'])

    # Define model architecture
    gen_shape = None
    out_dim = 1

    train_gen = p1b3.DataGenerator(loader, batch_size=gParameters['batch_size'], shape=gen_shape, name='train_gen', cell_noise_sigma=gParameters['cell_noise_sigma']).flow()
    val_gen = p1b3.DataGenerator(loader, partition='val', batch_size=gParameters['batch_size'], shape=gen_shape, name='val_gen').flow()
    # val_gen2 = p1b3.DataGenerator(loader, partition='val', batch_size=gParameters['batch_size'], shape=gen_shape, name='val_gen2').flow()
    # test_gen = p1b3.DataGenerator(loader, partition='test', batch_size=gParameters['batch_size'], shape=gen_shape, name='test_gen').flow()

    train_steps = int(loader.n_train/gParameters['batch_size'])
    val_steps = int(loader.n_val/gParameters['batch_size'])
    # test_steps = int(loader.n_test/gParameters['batch_size'])

    if 'train_steps' in gParameters:
        train_steps = gParameters['train_steps']
    if 'val_steps' in gParameters:
        val_steps = gParameters['val_steps']
    # if 'test_steps' in gParameters:
    #     test_steps = gParameters['test_steps']

    # e = np.array(train_gen)
    # e = next(train_gen)
    # print(f'type: {type(e)}')
    # print(f'len: {len(e)}')
    # print(f'len1: {len(e[0])}, len2: {len(e[1])}')
    # x = e[0]
    # y = e[1]
    # print(f'type_x: {type(x)}, type_y: {type(y)}')
    # print(f'shape_x: {np.shape(x)}, shape_y: {np.shape(y)}')
    print(f'train_steps: {train_steps}, val_steps: {val_steps}')

    return (train_gen, train_steps), (val_gen, val_steps)

if __name__ == '__main__':
    load_data()

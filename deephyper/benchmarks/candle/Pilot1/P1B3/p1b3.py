from __future__ import absolute_import

import collections
import gzip
import logging
import os
import sys
import multiprocessing
import threading
import argparse
try:
    import configparser
except ImportError:
    import ConfigParser as configparser

import numpy as np
import pandas as pd

from itertools import cycle, islice

from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler

file_path = os.path.dirname(os.path.realpath(__file__))
lib_path = os.path.abspath(os.path.join(file_path, '..', 'common'))
sys.path.append(lib_path)
lib_path2 = os.path.abspath(os.path.join(file_path, '..', '..', 'common'))
sys.path.append(lib_path2)

import p1_common


logger = logging.getLogger(__name__)

# Number of data generator workers
WORKERS = 1

np.set_printoptions(threshold=np.nan)

def common_parser(parser):

    parser.add_argument("--config_file", dest='config_file', type=str,
                        default=os.path.join(file_path, 'p1b3_default_model.txt'),
                        help="specify model configuration file")

    # Parse has been split between arguments that are common with the default neon parser
    # and all the other options
    parser = p1_common.get_default_neon_parse(parser)
    parser = p1_common.get_p1_common_parser(parser)

    # Arguments that are applicable just to p1b3
    parser = p1b3_parser(parser)

    return parser


def p1b3_parser(parser):

    # Feature selection
    parser.add_argument("--cell_features", nargs='+',
                        default=argparse.SUPPRESS,
                        choices=['expression', 'mirna', 'proteome', 'all', 'categorical'],
                        help="use one or more cell line feature sets: 'expression', 'mirna', 'proteome', 'all'; or use 'categorical' for one-hot encoding of cell lines")
    parser.add_argument("--drug_features", nargs='+',
                        default=argparse.SUPPRESS,
                        choices=['descriptors', 'latent', 'all', 'noise'],
                        help="use dragon7 descriptors, latent representations from Aspuru-Guzik's SMILES autoencoder, or both, or random features; 'descriptors','latent', 'all', 'noise'")
    parser.add_argument("--cell_noise_sigma", type=float,
                        help="standard deviation of guassian noise to add to cell line features during training")
    # Output selection
    parser.add_argument("--min_logconc", type=float,
                        default=argparse.SUPPRESS,
                        help="min log concentration of dose response data to use: -3.0 to -7.0")
    parser.add_argument("--max_logconc",  type=float,
                        default=argparse.SUPPRESS,
                        help="max log concentration of dose response data to use: -3.0 to -7.0")
    parser.add_argument("--subsample",
                        default=argparse.SUPPRESS,
                        choices=['naive_balancing', 'none'],
                        help="dose response subsample strategy; 'none' or 'naive_balancing'")
    parser.add_argument("--category_cutoffs", nargs='+', type=float,
                        default=argparse.SUPPRESS,
                        help="list of growth cutoffs (between -1 and +1) seperating non-response and response categories")
    # Sample data selection
    parser.add_argument("--test_cell_split", type=float,
                        default=argparse.SUPPRESS,
                        help="cell lines to use in test; if None use predefined unseen cell lines instead of sampling cell lines used in training")
    # Test random model
    parser.add_argument("--scramble", action="store_true",
                        default=False,
                        help="randomly shuffle dose response data")
    parser.add_argument("--workers", type=int,
                        default=WORKERS,
                        help="number of data generator workers")

    return parser

def read_config_file(file):
    config = configparser.ConfigParser()
    config.read(file)
    section = config.sections()
    fileParams = {}

    # default config values that we assume exists
    fileParams['activation']=eval(config.get(section[0],'activation'))
    fileParams['batch_size']=eval(config.get(section[0],'batch_size'))
    fileParams['batch_normalization']=eval(config.get(section[0],'batch_normalization'))
    fileParams['category_cutoffs']=eval(config.get(section[0],'category_cutoffs'))
    fileParams['cell_features']=eval(config.get(section[0],'cell_features'))
    fileParams['drop']=eval(config.get(section[0],'drop'))
    fileParams['drug_features']=eval(config.get(section[0],'drug_features'))
    fileParams['epochs']=eval(config.get(section[0],'epochs'))
    fileParams['feature_subsample']=eval(config.get(section[0],'feature_subsample'))
    fileParams['initialization']=eval(config.get(section[0],'initialization'))
    fileParams['learning_rate']=eval(config.get(section[0], 'learning_rate'))
    fileParams['loss']=eval(config.get(section[0],'loss'))
    fileParams['min_logconc']=eval(config.get(section[0],'min_logconc'))
    fileParams['max_logconc']=eval(config.get(section[0],'max_logconc'))
    fileParams['optimizer']=eval(config.get(section[0],'optimizer'))
#    fileParams['penalty']=eval(config.get(section[0],'penalty'))
    fileParams['rng_seed']=eval(config.get(section[0],'rng_seed'))
    fileParams['scaling']=eval(config.get(section[0],'scaling'))
    fileParams['subsample']=eval(config.get(section[0],'subsample'))
    fileParams['test_cell_split']=eval(config.get(section[0],'test_cell_split'))
    fileParams['validation_split']=eval(config.get(section[0],'validation_split'))
    fileParams['cell_noise_sigma']=eval(config.get(section[0],'cell_noise_sigma'))

    # parse the remaining values
    for k,v in config.items(section[0]):
        if not k in fileParams:
            fileParams[k] = eval(v)

    # Allow for either dense or convolutional layer specification
    # if none found exit
    try:
        fileParams['dense']=eval(config.get(section[0],'dense'))
    except configparser.NoOptionError:
        try:
            fileParams['conv']=eval(config.get(section[0],'conv'))
        except configparser.NoOptionError:
            print("Error ! No dense or conv layers specified. Wrong file !! ... exiting ")
            raise
        else:
            try:
                fileParams['pool']=eval(config.get(section[0],'pool'))
            except configparser.NoOptionError:
                fileParams['pool'] = None
                print("Warning ! No pooling specified after conv layer.")

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
    if 'cell_noise_sigma' in params:
        ext += '.N={}'.format(params['cell_noise_sigma'])
    if 'conv' in params:
        name = 'LC' if 'locally_connected' in params else 'C'
        layer_list = list(range(0, len(params['conv'])))
        for l, i in enumerate(layer_list):
            filters = params['conv'][i][0]
            filter_len = params['conv'][i][1]
            stride = params['conv'][i][2]
            if filters <= 0 or filter_len <= 0 or stride <= 0:
                break
            ext += '.{}{}={},{},{}'.format(name, l+1, filters, filter_len, stride)
        if 'pool' in params and params['conv'][0] and params['conv'][1]:
            ext += '.P={}'.format(params['pool'])
    if 'dense' in params:
        for i, n in enumerate(params['dense']):
            if n:
                ext += '.D{}={}'.format(i+1, n)
    if params['batch_normalization']:
        ext += '.BN'
    ext += '.S={}'.format(params['scaling'])

    return ext


def scale(df, scaling=None):
    """Scale data included in pandas dataframe.

    Parameters
    ----------
    df : pandas dataframe
        dataframe to scale
    scaling : 'maxabs', 'minmax', 'std', or None, optional (default 'std')
        type of scaling to apply
    """

    if scaling is None or scaling.lower() == 'none':
        return df

    df = df.dropna(axis=1, how='any')

    # Scaling data
    if scaling == 'maxabs':
        # Normalizing -1 to 1
        scaler = MaxAbsScaler()
    elif scaling == 'minmax':
        # Scaling to [0,1]
        scaler = MinMaxScaler()
    else:
        # Standard normalization
        scaler = StandardScaler()

    mat = df.as_matrix()
    mat = scaler.fit_transform(mat)

    df = pd.DataFrame(mat, columns=df.columns)

    return df


def impute_and_scale(df, scaling='std'):
    """Impute missing values with mean and scale data included in pandas dataframe.

    Parameters
    ----------
    df : pandas dataframe
        dataframe to impute and scale
    scaling : 'maxabs' [-1,1], 'minmax' [0,1], 'std', or None, optional (default 'std')
        type of scaling to apply
    """

    df = df.dropna(axis=1, how='all')

    imputer = Imputer(strategy='mean', axis=0)
    mat = imputer.fit_transform(df)

    if scaling is None or scaling.lower() == 'none':
        return pd.DataFrame(mat, columns=df.columns)

    if scaling == 'maxabs':
        scaler = MaxAbsScaler()
    elif scaling == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()

    mat = scaler.fit_transform(mat)

    df = pd.DataFrame(mat, columns=df.columns)

    return df


def load_cellline_expressions(path, dtype, ncols=None, scaling='std'):
    """Load cell line expression data, sub-select columns of gene expression
        randomly if specificed, scale the selected data and return a
        pandas dataframe.

    Parameters
    ----------
    path: string
        path to 'RNA_5_Platform_Gene_Transcript_Averaged_intensities.transposed.txt'
    dtype: numpy type
        precision (data type) for reading float values
    ncols : int or None
        number of columns (gene expression) to randomly subselect (default None : use all data)
    scaling : 'maxabs' [-1,1], 'minmax' [0,1], 'std', or None, optional (default 'std')
        type of scaling to apply
    """

    df = pd.read_csv(path, sep='\t', engine='c',
                     na_values=['na','-',''])

    df1 = df['CellLine']
    df1 = df1.map(lambda x: x.replace('.', ':'))
    df1.name = 'CELLNAME'

    df2 = df.drop('CellLine', 1)

    total = df2.shape[1]
    if ncols and ncols < total:
        usecols = np.random.choice(total, size=ncols, replace=False)
        df2 = df2.iloc[:, usecols]

    df2 = impute_and_scale(df2, scaling)
    df2 = df2.astype(dtype)
    df = pd.concat([df1, df2], axis=1)

    return df


def load_cellline_mirna(path, dtype, ncols=None, scaling='std'):
    """Load cell line microRNA data, sub-select columns randomly if
        specificed, scale the selected data and return a pandas
        dataframe.

    Parameters
    ----------
    path: string
        path to 'RNA__microRNA_OSU_V3_chip_log2.transposed.txt'
    dtype: numpy type
        precision (data type) for reading float values
    ncols : int or None
        number of columns to randomly subselect (default None : use all data)
    scaling : 'maxabs' [-1,1], 'minmax' [0,1], 'std', or None, optional (default 'std')
        type of scaling to apply

    """

    df = pd.read_csv(path, sep='\t', engine='c',
                     na_values=['na','-',''])

    df1 = df['CellLine']
    df1 = df1.map(lambda x: x.replace('.', ':'))
    df1.name = 'CELLNAME'

    df2 = df.drop('CellLine', 1)

    total = df2.shape[1]
    if ncols and ncols < total:
        usecols = np.random.choice(total, size=ncols, replace=False)
        df2 = df2.iloc[:, usecols]

    df2 = impute_and_scale(df2, scaling)
    df2 = df2.astype(dtype)
    df = pd.concat([df1, df2], axis=1)

    return df


def load_cellline_proteome(path, dtype, kinome_path=None, ncols=None, scaling='std'):
    """Load cell line microRNA data, sub-select columns randomly if
        specificed, scale the selected data and return a pandas
        dataframe.

    Parameters
    ----------
    path: string
        path to 'nci60_proteome_log2.transposed.tsv'
    dtype: numpy type
        precision (data type) for reading float values
    kinome_path: string or None (default None)
        path to 'nci60_kinome_log2.transposed.tsv'
    ncols : int or None
        number of columns to randomly subselect (default None : use all data)
    scaling : 'maxabs' [-1,1], 'minmax' [0,1], 'std', or None, optional (default 'std')
        type of scaling to apply

    """

    df = pd.read_csv(path, sep='\t', engine='c')
    df = df.set_index('CellLine')

    if kinome_path:
        df_k = pd.read_csv(kinome_path, sep='\t', engine='c')
        df_k = df_k.set_index('CellLine')
        df_k = df_k.add_suffix('.K')
        df = df.merge(df_k, left_index=True, right_index=True)

    index = df.index.map(lambda x: x.replace('.', ':'))

    total = df.shape[1]
    if ncols and ncols < total:
        usecols = np.random.choice(total, size=ncols, replace=False)
        df = df.iloc[:, usecols]

    df = impute_and_scale(df, scaling)
    df = df.astype(dtype)

    df.index = index
    df.index.names = ['CELLNAME']
    df = df.reset_index()

    return df


def load_drug_descriptors(path, dtype, ncols=None, scaling='std'):
    """Load drug descriptor data, sub-select columns of drugs descriptors
        randomly if specificed, impute and scale the selected data, and return a
        pandas dataframe.

    Parameters
    ----------
    path: string
        path to 'descriptors.2D-NSC.5dose.filtered.txt'
    dtype: numpy type
        precision (data type) for reading float values
    ncols : int or None
        number of columns (drugs descriptors) to randomly subselect (default None : use all data)
    scaling : 'maxabs' [-1,1], 'minmax' [0,1], 'std', or None, optional (default 'std')
        type of scaling to apply
    """

    df = pd.read_csv(path, sep='\t', engine='c',
                     na_values=['na','-',''],
                     dtype=dtype,
                     converters ={'NAME' : str})

    df1 = pd.DataFrame(df.loc[:,'NAME'])
    df1.rename(columns={'NAME': 'NSC'}, inplace=True)

    df2 = df.drop('NAME', 1)

    # # Filter columns if requested

    total = df2.shape[1]
    if ncols and ncols < total:
        usecols = np.random.choice(total, size=ncols, replace=False)
        df2 = df2.iloc[:,usecols]

    df2 = impute_and_scale(df2, scaling)
    df2 = df2.astype(dtype)

    df_dg = pd.concat([df1, df2], axis=1)

    return df_dg


def load_drug_autoencoded(path, dtype, ncols=None, scaling='std'):
    """Load drug latent representation from autoencoder, sub-select
    columns of drugs randomly if specificed, impute and scale the
    selected data, and return a pandas dataframe.

    Parameters
    ----------
    path: string
        path to 'Aspuru-Guzik_NSC_latent_representation_292D.csv'
    dtype: numpy type
        precision (data type) for reading float values
    ncols : int or None
        number of columns (drug latent representations) to randomly subselect (default None : use all data)
    scaling : 'maxabs' [-1,1], 'minmax' [0,1], 'std', or None, optional (default 'std')
        type of scaling to apply

    """

    df = pd.read_csv(path, engine='c', converters ={'NSC' : str}, dtype=dtype)

    df1 = pd.DataFrame(df.loc[:, 'NSC'])
    df2 = df.drop('NSC', 1)

    total = df2.shape[1]
    if ncols and ncols < total:
        usecols = np.random.choice(total, size=ncols, replace=False)
        df2 = df2.iloc[:, usecols]

    df2 = impute_and_scale(df2, scaling)
    df2 = df2.astype(dtype)

    df = pd.concat([df1, df2], axis=1)

    return df


def load_dose_response(path, seed, dtype, min_logconc=-5., max_logconc=-5., subsample=None):
    """Load cell line response to different drug compounds, sub-select response for a specific
        drug log concentration range and return a pandas dataframe.

    Parameters
    ----------
    path: string
        path to 'NCI60_dose_response_with_missing_z5_avg.csv'
    seed: integer
        seed for random generation
    dtype: numpy type
        precision (data type) for reading float values
    min_logconc : -3, -4, -5, -6, -7, optional (default -5)
        min log concentration of drug to return cell line growth
    max_logconc : -3, -4, -5, -6, -7, optional (default -5)
        max log concentration of drug to return cell line growth
    subsample: None, 'naive_balancing' (default None)
        subsampling strategy to use to balance the data based on growth
    """

    df = pd.read_csv(path, sep=',', engine='c',
                     na_values=['na','-',''],
                     dtype={'NSC':object, 'CELLNAME':str, 'LOG_CONCENTRATION':dtype, 'GROWTH':dtype})

    df = df[(df['LOG_CONCENTRATION'] >= min_logconc) & (df['LOG_CONCENTRATION'] <= max_logconc)]

    df = df[['NSC', 'CELLNAME', 'GROWTH', 'LOG_CONCENTRATION']]

    if subsample and subsample == 'naive_balancing':
        df1 = df[df['GROWTH'] <= 0]
        df2 = df[(df['GROWTH'] > 0) & (df['GROWTH'] < 50)].sample(frac=0.7, random_state=seed)
        df3 = df[(df['GROWTH'] >= 50) & (df['GROWTH'] <= 100)].sample(frac=0.18, random_state=seed)
        df4 = df[df['GROWTH'] > 100].sample(frac=0.01, random_state=seed)
        df = pd.concat([df1, df2, df3, df4])

    df = df.set_index(['NSC'])

    return df

def stage_data():
    server = 'http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/P1B3/'

    cell_expr_path = p1_common.get_p1_file(server+'P1B3_cellline_expressions.tsv')
    cell_mrna_path = p1_common.get_p1_file(server+'P1B3_cellline_mirna.tsv')
    cell_prot_path = p1_common.get_p1_file(server+'P1B3_cellline_proteome.tsv')
    cell_kino_path = p1_common.get_p1_file(server+'P1B3_cellline_kinome.tsv')
    drug_desc_path = p1_common.get_p1_file(server+'P1B3_drug_descriptors.tsv')
    drug_auen_path = p1_common.get_p1_file(server+'P1B3_drug_latent.csv')
    dose_resp_path = p1_common.get_p1_file(server+'P1B3_dose_response.csv')
    test_cell_path = p1_common.get_p1_file(server+'P1B3_test_celllines.txt')
    test_drug_path = p1_common.get_p1_file(server+'P1B3_test_drugs.txt')

    return(cell_expr_path, cell_mrna_path, cell_prot_path, cell_kino_path,
           drug_desc_path, drug_auen_path, dose_resp_path, test_cell_path,
           test_drug_path)

class DataLoader(object):
    """Load merged drug response, drug descriptors and cell line essay data
    """

    def __init__(self, seed, dtype, val_split=0.2, test_cell_split=None, shuffle=True,
                 cell_features=['expression'], drug_features=['descriptors'],
                 feature_subsample=None, scaling='std', scramble=False,
                 min_logconc=-5., max_logconc=-4., subsample='naive_balancing',
                 category_cutoffs=[0.]):
        """Initialize data merging drug response, drug descriptors and cell line essay.
           Shuffle and split training and validation set

        Parameters
        ----------
        seed: integer
            seed for random generation
        dtype: numpy type
            precision (data type) for reading float values
        val_split : float, optional (default 0.2)
            fraction of data to use in validation
        test_cell_split : float or None, optional (default None)
            fraction of cell lines to use in test; if None use predefined unseen cell lines instead of sampling cell lines used in training
        shuffle : True or False, optional (default True)
            if True shuffles the merged data before splitting training and validation sets
        cell_features: list of strings from 'expression', 'mirna', 'proteome', 'all', 'categorical' (default ['expression'])
            use one or more cell line feature sets: gene expression, microRNA, proteomics; or, use 'categorical' for one-hot encoded cell lines
        drug_features: list of strings from 'descriptors', 'latent', 'all', 'noise' (default ['descriptors'])
            use dragon7 descriptors, latent representations from Aspuru-Guzik's SMILES autoencoder trained on NSC drugs, or both; use random features if set to noise
        feature_subsample: None or integer (default None)
            number of feature columns to use from cellline expressions and drug descriptors
        scaling: None, 'std', 'minmax' or 'maxabs' (default 'std')
            type of feature scaling: 'maxabs' to [-1,1], 'maxabs' to [-1, 1], 'std' for standard normalization
        scramble: True or False, optional (default False)
            if True randomly shuffle dose response data as a control
        min_logconc: float value between -3 and -7, optional (default -5.)
            min log concentration of drug to return cell line growth
        max_logconc: float value between -3 and -7, optional (default -4.)
            max log concentration of drug to return cell line growth
        subsample: 'naive_balancing' or None
            if True balance dose response data with crude subsampling
        category_cutoffs: list of floats (between -1 and +1) (default None)
            growth thresholds seperating non-response and response categories
        """

        cell_expr_path, cell_mrna_path, cell_prot_path, cell_kino_path,drug_desc_path, drug_auen_path, dose_resp_path, test_cell_path, test_drug_path = stage_data()
        # Seed random generator for loading data
        np.random.seed(seed)

        df = load_dose_response(dose_resp_path, seed, dtype,
                                min_logconc=min_logconc, max_logconc=max_logconc, subsample=subsample)
        logger.info('Loaded {} unique (D, CL) response sets.'.format(df.shape[0]))
        # df[['GROWTH', 'LOG_CONCENTRATION']].to_csv('all.response.csv')
        df = df.reset_index()

        if 'all' in cell_features:
            self.cell_features = ['expression', 'mirna', 'proteome']
        else:
            self.cell_features = cell_features

        if 'all' in drug_features:
            self.drug_features = ['descriptors', 'latent']
        else:
            self.drug_features = drug_features

        self.input_shapes = collections.OrderedDict()
        self.input_shapes['drug_concentration'] = (1,)

        for fea in self.cell_features:
            if fea == 'expression':
                self.df_cell_expr = load_cellline_expressions(cell_expr_path, dtype, ncols=feature_subsample, scaling=scaling)
                self.input_shapes['cell_expression'] = (self.df_cell_expr.shape[1] - 1,)
                df = df.merge(self.df_cell_expr[['CELLNAME']], on='CELLNAME')
            elif fea == 'mirna':
                self.df_cell_mirna = load_cellline_mirna(cell_mrna_path, dtype, ncols=feature_subsample, scaling=scaling)
                self.input_shapes['cell_microRNA'] = (self.df_cell_mirna.shape[1] - 1,)
                df = df.merge(self.df_cell_mirna[['CELLNAME']], on='CELLNAME')
            elif fea == 'proteome':
                self.df_cell_prot = load_cellline_proteome(cell_prot_path, dtype, cell_kino_path, ncols=feature_subsample, scaling=scaling)
                self.input_shapes['cell_proteome'] = (self.df_cell_prot.shape[1] - 1,)
                df = df.merge(self.df_cell_prot[['CELLNAME']], on='CELLNAME')
            elif fea == 'categorical':
                df_cell_ids = df[['CELLNAME']].drop_duplicates()
                cell_ids = df_cell_ids['CELLNAME'].map(lambda x: x.replace(':', '.'))
                df_cell_cat = pd.get_dummies(cell_ids)
                df_cell_cat.index = df_cell_ids['CELLNAME']
                self.df_cell_cat = df_cell_cat.reset_index()
                self.input_shapes['cell_categorical'] = (self.df_cell_cat.shape[1] - 1,)

        for fea in self.drug_features:
            if fea == 'descriptors':
                self.df_drug_desc = load_drug_descriptors(drug_desc_path, dtype, ncols=feature_subsample, scaling=scaling)
                self.input_shapes['drug_descriptors'] = (self.df_drug_desc.shape[1] - 1,)
                df = df.merge(self.df_drug_desc[['NSC']], on='NSC')
            elif fea == 'latent':
                self.df_drug_auen = load_drug_autoencoded(drug_auen_path, dtype, ncols=feature_subsample, scaling=scaling)
                self.input_shapes['drug_SMILES_latent'] = (self.df_drug_auen.shape[1] - 1,)
                df = df.merge(self.df_drug_auen[['NSC']], on='NSC')
            elif fea == 'noise':
                df_drug_ids = df[['NSC']].drop_duplicates()
                noise = np.random.normal(size=(df_drug_ids.shape[0], 500))
                df_rand = pd.DataFrame(noise, index=df_drug_ids['NSC'],
                                       columns=['RAND-{:03d}'.format(x) for x in range(500)])
                self.df_drug_rand = df_rand.reset_index()
                self.input_shapes['drug_random_vector'] = (self.df_drug_rand.shape[1] - 1,)

        logger.debug('Filtered down to {} rows with matching information.'.format(df.shape[0]))
        # df[['GROWTH', 'LOG_CONCENTRATION']].to_csv('filtered.response.csv')

        df_test_cell = pd.read_csv(test_cell_path)
        df_test_drug = pd.read_csv(test_drug_path, dtype={'NSC':object})

        df_train_val = df[(~df['NSC'].isin(df_test_drug['NSC'])) & (~df['CELLNAME'].isin(df_test_cell['CELLNAME']))]
        logger.debug('Combined train and validation set has {} rows'.format(df_train_val.shape[0]))

        if test_cell_split and test_cell_split > 0:
            df_test_cell = df_train_val[['CELLNAME']].drop_duplicates().sample(frac=test_cell_split, random_state=seed)
            logger.debug('Use unseen drugs and a fraction of seen cell lines for testing: ' + ', '.join(sorted(list(df_test_cell['CELLNAME']))))
        else:
            logger.debug('Use unseen drugs and predefined unseen cell lines for testing: ' + ', '.join(sorted(list(df_test_cell['CELLNAME']))))

        df_test = df.merge(df_test_cell, on='CELLNAME').merge(df_test_drug, on='NSC')
        logger.debug('Test set has {} rows'.format(df_test.shape[0]))

        if shuffle:
            df_train_val = df_train_val.sample(frac=1.0, random_state=seed)
            df_test = df_test.sample(frac=1.0, random_state=seed)

        self.df_response = pd.concat([df_train_val, df_test]).reset_index(drop=True)

        if scramble:
            growth = self.df_response[['GROWTH']]
            random_growth = growth.iloc[np.random.permutation(np.arange(growth.shape[0]))].reset_index()
            self.df_response[['GROWTH']] = random_growth['GROWTH']
            logger.warn('Randomly shuffled dose response growth values.')

        logger.info('Distribution of dose response:')
        logger.info(self.df_response[['GROWTH']].describe())

        if category_cutoffs is not None:
            growth = self.df_response['GROWTH']
            classes = np.digitize(growth, category_cutoffs)
            bc = np.bincount(classes)
            min_g = np.min(growth) / 100
            max_g = np.max(growth) / 100
            logger.info('Category cutoffs: {}'.format(category_cutoffs))
            logger.info('Dose response bin counts:')
            for i, count in enumerate(bc):
                lower = min_g if i == 0 else category_cutoffs[i-1]
                upper = max_g if i == len(bc)-1 else category_cutoffs[i]
                logger.info('  Class {}: {:7d} ({:.4f}) - between {:+.2f} and {:+.2f}'.
                            format(i, count, count/len(growth), lower, upper))
            logger.info('  Total: {:9d}'.format(len(growth)))

        self.total = df_train_val.shape[0]
        self.n_test = df_test.shape[0]
        self.n_val = int(self.total * val_split)
        self.n_train = self.total - self.n_val
        logger.info('Rows in train: {}, val: {}, test: {}'.format(self.n_train, self.n_val, self.n_test))

        logger.info('Input features shapes:')
        for k, v in self.input_shapes.items():
            logger.info('  {}: {}'.format(k, v))

        self.input_dim = sum([np.prod(x) for x in self.input_shapes.values()])
        logger.info('Total input dimensions: {}'.format(self.input_dim))


class DataGenerator(object):
    """Generate training, validation or testing batches from loaded data
    """

    def __init__(self, data, partition='train', batch_size=32, shape=None, concat=True, name='', cell_noise_sigma=None):
        """Initialize data

        Parameters
        ----------
        data: DataLoader object
            loaded data object containing original data frames for molecular, drug and response data
        partition: 'train', 'val', or 'test'
            partition of data to generate for
        batch_size: integer (default 32)
            batch size of generated data
        shape: None, '1d' or 'add_1d' (default None)
            keep original feature shapes, make them flat or add one extra dimension (for convolution or locally connected layers in some frameworks)
        concat: True or False (default True)
            concatenate all features if set to True
        cell_noise_sigma: float
            standard deviation of guassian noise to add to cell line features during training
        """
        self.lock = threading.Lock()
        self.data = data
        self.partition = partition
        self.batch_size = batch_size
        self.shape = shape
        self.concat = concat
        self.name = name
        self.cell_noise_sigma = cell_noise_sigma

        if partition == 'train':
            self.cycle = cycle(range(data.n_train))
            self.num_data = data.n_train
        elif partition == 'val':
            self.cycle = cycle(range(data.total)[-data.n_val:])
            self.num_data = data.n_val
        elif partition == 'test':
            self.cycle = cycle(range(data.total, data.total + data.n_test))
            self.num_data = data.n_test
        else:
            raise Exception('Data partition "{}" not recognized.'.format(partition))

    def flow(self):
        """Keep generating data batches
        """
        while 1:
            self.lock.acquire()
            indices = list(islice(self.cycle, self.batch_size))
            # print("\nProcess: {}, Batch indices start: {}".format(multiprocessing.current_process().name, indices[0]))
            # logger.debug('Gen {} at index: {}'.format(self.name, indices[0]))
            self.lock.release()

            df = self.data.df_response.iloc[indices, :]
            cell_column_beg = df.shape[1]

            for fea in self.data.cell_features:
                if fea == 'expression':
                    df = pd.merge(df, self.data.df_cell_expr, on='CELLNAME')
                elif fea == 'mirna':
                    df = pd.merge(df, self.data.df_cell_mirna, on='CELLNAME')
                elif fea == 'proteome':
                    df = pd.merge(df, self.data.df_cell_prot, on='CELLNAME')
                elif fea == 'categorical':
                    df = pd.merge(df, self.data.df_cell_cat, on='CELLNAME')

            cell_column_end = df.shape[1]

            for fea in self.data.drug_features:
                if fea == 'descriptors':
                    df = df.merge(self.data.df_drug_desc, on='NSC')
                elif fea == 'latent':
                    df = df.merge(self.data.df_drug_auen, on='NSC')
                elif fea == 'noise':
                    df = df.merge(self.data.df_drug_rand, on='NSC')

            df = df.drop(['CELLNAME', 'NSC'], 1)
            x = np.array(df.iloc[:, 1:])

            if self.cell_noise_sigma:
                c1 = cell_column_beg - 3
                c2 = cell_column_end - 3
                x[:, c1:c2] += np.random.randn(df.shape[0], c2-c1) * self.cell_noise_sigma

            y = np.array(df.iloc[:, 0])
            y = y / 100.

            if self.concat:
                if self.shape == 'add_1d':
                    yield x.reshape(x.shape + (1,)), y
                else:
                    yield x, y
            else:
                x_list = []
                index = 0
                for v in self.data.input_shapes.values():
                    length = np.prod(v)
                    subset = x[:, index:index+length]
                    if self.shape == '1d':
                        reshape = (x.shape[0], length)
                    elif self.shape == 'add_1d':
                        reshape = (x.shape[0],) + v + (1,)
                    else:
                        reshape = (x.shape[0],) + v
                    x_list.append(subset.reshape(reshape))
                    index += length
                yield x_list, y

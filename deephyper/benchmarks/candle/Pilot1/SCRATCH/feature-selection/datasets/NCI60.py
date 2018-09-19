from __future__ import print_function

import collections
import os
import re
import sys

import numpy as np
import pandas as pd

from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler

file_path = os.path.dirname(os.path.realpath(__file__))
lib_path = os.path.abspath(os.path.join(file_path, '..', 'utils'))
sys.path.append(lib_path)

from data_utils import get_file


global_cache = {}

SEED = 2017
P1B3_URL = 'http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/P1B3/'


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


def describe_response_data(df, cells=['all'], drugs=['A'], doses=[-5, -4]):
    if 'all' in cells or cells == 'all':
        cells = all_cells()
    if 'all' in drugs or drugs == 'all':
        drugs = all_drugs()
    elif len(drugs) == 1 and re.match("^[ABC]$", drugs[0].upper()):
        drugs = drugs_in_set('Jason:' + drugs[0].upper())

    print('cells:', cells)
    print('drugs:', drugs)

    lconc = -4
    for cell in cells:
        d = df[(df['CELLNAME'] == cell) & (df['LOG_CONCENTRATION'] == lconc)]
        print(cell)
        print(d.describe())
        break


def load_dose_response(min_logconc=-4., max_logconc=-4., subsample=None, fraction=False):
    """Load cell line response to different drug compounds, sub-select response for a specific
        drug log concentration range and return a pandas dataframe.

    Parameters
    ----------
    min_logconc : -3, -4, -5, -6, -7, optional (default -4)
        min log concentration of drug to return cell line growth
    max_logconc : -3, -4, -5, -6, -7, optional (default -4)
        max log concentration of drug to return cell line growth
    subsample: None, 'naive_balancing' (default None)
        subsampling strategy to use to balance the data based on growth
    fraction: bool (default False)
        divide growth percentage by 100
    """

    path = get_file(P1B3_URL + 'NCI60_dose_response_with_missing_z5_avg.csv')

    df = global_cache.get(path)
    if df is None:
        df = pd.read_csv(path, sep=',', engine='c',
                         na_values=['na', '-', ''],
                         dtype={'NSC':object, 'CELLNAME':str, 'LOG_CONCENTRATION':np.float32, 'GROWTH':np.float32})
        global_cache[path] = df

    df = df[(df['LOG_CONCENTRATION'] >= min_logconc) & (df['LOG_CONCENTRATION'] <= max_logconc)]

    df = df[['NSC', 'CELLNAME', 'GROWTH', 'LOG_CONCENTRATION']]

    if subsample and subsample == 'naive_balancing':
        df1 = df[df['GROWTH'] <= 0]
        df2 = df[(df['GROWTH'] > 0) & (df['GROWTH'] < 50)].sample(frac=0.7, random_state=SEED)
        df3 = df[(df['GROWTH'] >= 50) & (df['GROWTH'] <= 100)].sample(frac=0.18, random_state=SEED)
        df4 = df[df['GROWTH'] > 100].sample(frac=0.01, random_state=SEED)
        df = pd.concat([df1, df2, df3, df4])

    if fraction:
        df['GROWTH'] /= 100

    df = df.set_index(['NSC'])

    return df


def load_drug_descriptors(ncols=None, scaling='std', add_prefix=True):
    """Load drug descriptor data, sub-select columns of drugs descriptors
        randomly if specificed, impute and scale the selected data, and return a
        pandas dataframe.

    Parameters
    ----------
    ncols : int or None
        number of columns (drugs descriptors) to randomly subselect (default None : use all data)
    scaling : 'maxabs' [-1,1], 'minmax' [0,1], 'std', or None, optional (default 'std')
        type of scaling to apply
    add_prefix: True or False
        add feature namespace prefix
    """

    path = get_file(P1B3_URL + 'descriptors.2D-NSC.5dose.filtered.txt')

    df = global_cache.get(path)
    if df is None:
        df = pd.read_csv(path, sep='\t', engine='c',
                         na_values=['na','-',''],
                         dtype=np.float32)
        global_cache[path] = df

    df1 = pd.DataFrame(df.loc[:,'NAME'].astype(int).astype(str))
    df1.rename(columns={'NAME': 'NSC'}, inplace=True)

    df2 = df.drop('NAME', 1)
    if add_prefix:
        df2 = df2.add_prefix('dragon7.')

    total = df2.shape[1]
    if ncols and ncols < total:
        usecols = np.random.choice(total, size=ncols, replace=False)
        df2 = df2.iloc[:,usecols]

    df2 = impute_and_scale(df2, scaling)
    df2 = df2.astype(np.float32)

    df_dg = pd.concat([df1, df2], axis=1)

    return df_dg


def load_cell_expression_u133p2(ncols=None, scaling='std', add_prefix=True):
    """Load U133_Plus2 cell line expression data prepared by Judith,
        sub-select columns of gene expression randomly if specificed,
        scale the selected data and return a pandas dataframe.

    Parameters
    ----------
    ncols : int or None
        number of columns (gene expression) to randomly subselect (default None : use all data)
    scaling : 'maxabs' [-1,1], 'minmax' [0,1], 'std', or None, optional (default 'std')
        type of scaling to apply
    add_prefix: True or False
        add feature namespace prefix
    """
    path = get_file('http://bioseed.mcs.anl.gov/~fangfang/p1h/GSE32474_U133Plus2_GCRMA_gene_median.txt')

    df = global_cache.get(path)
    if df is None:
        df = pd.read_csv(path, sep='\t', engine='c')
        global_cache[path] = df

    df1 = df['CELLNAME']
    df2 = df.drop('CELLNAME', 1)
    if add_prefix:
        df2 = df2.add_prefix('expr.')

    total = df.shape[1]
    if ncols and ncols < total:
        usecols = np.random.choice(total, size=ncols, replace=False)
        df2 = df2.iloc[:, usecols]

    df2 = impute_and_scale(df2, scaling)
    df2 = df2.astype(np.float32)
    df = pd.concat([df1, df2], axis=1)

    return df


def load_cell_expression_5platform(ncols=None, scaling='std', add_prefix=True):
    """Load 5-platform averaged cell line expression data, sub-select
        columns of gene expression randomly if specificed, scale the
        selected data and return a pandas dataframe.

    Parameters
    ----------
    ncols : int or None
        number of columns (gene expression) to randomly subselect (default None : use all data)
    scaling : 'maxabs' [-1,1], 'minmax' [0,1], 'std', or None, optional (default 'std')
        type of scaling to apply
    add_prefix: True or False
        add feature namespace prefix
    """

    path = get_file(P1B3_URL + 'RNA_5_Platform_Gene_Transcript_Averaged_intensities.transposed.txt')

    df = global_cache.get(path)
    if df is None:
        df = pd.read_csv(path, sep='\t', engine='c',
                         na_values=['na','-',''])
        global_cache[path] = df

    df1 = df['CellLine']
    df1 = df1.map(lambda x: x.replace('.', ':'))
    df1.name = 'CELLNAME'

    df2 = df.drop('CellLine', 1)
    if add_prefix:
        df2 = df2.add_prefix('expr_5p.')

    total = df2.shape[1]
    if ncols and ncols < total:
        usecols = np.random.choice(total, size=ncols, replace=False)
        df2 = df2.iloc[:, usecols]

    df2 = impute_and_scale(df2, scaling)
    df2 = df2.astype(np.float32)
    df = pd.concat([df1, df2], axis=1)

    return df


def load_cell_mirna(ncols=None, scaling='std', add_prefix=True):
    """Load cell line microRNA data, sub-select columns randomly if
        specificed, scale the selected data and return a pandas
        dataframe.

    Parameters
    ----------
    ncols : int or None
        number of columns to randomly subselect (default None : use all data)
    scaling : 'maxabs' [-1,1], 'minmax' [0,1], 'std', or None, optional (default 'std')
        type of scaling to apply
    add_prefix: True or False
        add feature namespace prefix
    """
    path = get_file(P1B3_URL + 'RNA__microRNA_OSU_V3_chip_log2.transposed.txt')

    df = global_cache.get(path)
    if df is None:
        df = pd.read_csv(path, sep='\t', engine='c',
                         na_values=['na','-',''])
        global_cache[path] = df

    df1 = df['CellLine']
    df1 = df1.map(lambda x: x.replace('.', ':'))
    df1.name = 'CELLNAME'

    df2 = df.drop('CellLine', 1)
    if add_prefix:
        df2 = df2.add_prefix('mRNA.')

    total = df2.shape[1]
    if ncols and ncols < total:
        usecols = np.random.choice(total, size=ncols, replace=False)
        df2 = df2.iloc[:, usecols]

    df2 = impute_and_scale(df2, scaling)
    df2 = df2.astype(np.float32)
    df = pd.concat([df1, df2], axis=1)

    return df


def load_cell_proteome(ncols=None, scaling='std', add_prefix=True):
    """Load cell line microRNA data, sub-select columns randomly if
        specificed, scale the selected data and return a pandas
        dataframe.

    Parameters
    ----------
    ncols : int or None
        number of columns to randomly subselect (default None : use all data)
    scaling : 'maxabs' [-1,1], 'minmax' [0,1], 'std', or None, optional (default 'std')
        type of scaling to apply
    add_prefix: True or False
        add feature namespace prefix
    """

    path1 = get_file(P1B3_URL + 'nci60_proteome_log2.transposed.tsv')
    path2 = get_file(P1B3_URL + 'nci60_kinome_log2.transposed.tsv')

    df = global_cache.get(path1)
    if df is None:
        df = pd.read_csv(path1, sep='\t', engine='c')
        global_cache[path1] = df

    df_k = global_cache.get(path2)
    if df_k is None:
        df_k = pd.read_csv(path2, sep='\t', engine='c')
        global_cache[path2] = df_k

    df = df.set_index('CellLine')
    df_k = df_k.set_index('CellLine')

    if add_prefix:
        df = df.add_prefix('prot.')
        df_k = df_k.add_prefix('kino.')
    else:
        df_k = df_k.add_suffix('.K')

    df = df.merge(df_k, left_index=True, right_index=True)

    index = df.index.map(lambda x: x.replace('.', ':'))

    total = df.shape[1]
    if ncols and ncols < total:
        usecols = np.random.choice(total, size=ncols, replace=False)
        df = df.iloc[:, usecols]

    df = impute_and_scale(df, scaling)
    df = df.astype(np.float32)

    df.index = index
    df.index.names = ['CELLNAME']
    df = df.reset_index()

    return df


def load_drug_autoencoded_AG(ncols=None, scaling='std', add_prefix=True):
    """Load drug latent representation from Aspuru-Guzik's variational
    autoencoder, sub-select columns of drugs randomly if specificed,
    impute and scale the selected data, and return a pandas dataframe

    Parameters
    ----------
    ncols : int or None
        number of columns (drug latent representations) to randomly subselect (default None : use all data)
    scaling : 'maxabs' [-1,1], 'minmax' [0,1], 'std', or None, optional (default 'std')
        type of scaling to apply
    add_prefix: True or False
        add feature namespace prefix
    """
    path = get_file(P1B3_URL + 'Aspuru-Guzik_NSC_latent_representation_292D.csv')

    df = global_cache.get(path)
    if df is None:
        df = pd.read_csv(path, engine='c', dtype=np.float32)
        global_cache[path] = df

    df1 = pd.DataFrame(df.loc[:, 'NSC'].astype(int).astype(str))
    df2 = df.drop('NSC', 1)
    if add_prefix:
        df2 = df2.add_prefix('smiles_latent_AG.')

    total = df2.shape[1]
    if ncols and ncols < total:
        usecols = np.random.choice(total, size=ncols, replace=False)
        df2 = df2.iloc[:, usecols]

    df2 = impute_and_scale(df2, scaling)
    df2 = df2.astype(np.float32)

    df = pd.concat([df1, df2], axis=1)

    return df


def all_cells():
    df = load_dose_response()
    return df['CELLNAME'].drop_duplicates().tolist()


def all_drugs():
    df = load_dose_response()
    return df['NSC'].drop_duplicates().tolist()


def drugs_in_set(set_name):
    path = get_file('http://bioseed.mcs.anl.gov/~fangfang/p1h/NCI60_drug_sets.tsv')
    df = pd.read_csv(path, sep='\t', engine='c')
    drugs = df[df['Drug_Set'] == set_name].iloc[0][1].split(',')
    return drugs


def load_by_cell_data(cell='BR:MCF7', drug_features=['descriptors'], shuffle=True,
                      min_logconc=-5., max_logconc=-4., subsample='naive_balancing',
                      feature_subsample=None, scaling='std', scramble=False, verbose=True):

    """Load dataframe for by cellline models

    Parameters
    ----------
    cell: cellline ID
    drug_features: list of strings from 'descriptors', 'latent', 'all', 'noise' (default ['descriptors'])
        use dragon7 descriptors, latent representations from Aspuru-Guzik's SMILES autoencoder
        trained on NSC drugs, or both; use random features if set to noise
    shuffle : True or False, optional (default True)
        if True shuffles the merged data before splitting training and validation sets
    scramble: True or False, optional (default False)
        if True randomly shuffle dose response data as a control
    min_logconc: float value between -3 and -7, optional (default -5.)
        min log concentration of drug to return cell line growth
    max_logconc: float value between -3 and -7, optional (default -4.)
        max log concentration of drug to return cell line growth
    feature_subsample: None or integer (default None)
        number of feature columns to use from cellline expressions and drug descriptors
    scaling: None, 'std', 'minmax' or 'maxabs' (default 'std')
        type of feature scaling: 'maxabs' to [-1,1], 'maxabs' to [-1, 1], 'std' for standard normalization
    subsample: 'naive_balancing' or None
        if True balance dose response data with crude subsampling
    scramble: True or False, optional (default False)
        if True randomly shuffle dose response data as a control
    """

    if 'all' in drug_features:
        drug_features = ['descriptors', 'latent']

    df_resp = load_dose_response(subsample=subsample, min_logconc=min_logconc, max_logconc=max_logconc, fraction=True)

    df = df_resp[df_resp['CELLNAME'] == cell].reset_index()
    df = df[['NSC', 'GROWTH', 'LOG_CONCENTRATION']]
    df = df.rename(columns={'LOG_CONCENTRATION': 'LCONC'})

    input_dims = collections.OrderedDict()
    input_dims['log_conc'] = 1

    for fea in drug_features:
        if fea == 'descriptors':
            df_desc = load_drug_descriptors(ncols=feature_subsample, scaling=scaling)
            df = df.merge(df_desc, on='NSC')
            input_dims['drug_descriptors'] = df_desc.shape[1] - 1
        elif fea == 'latent':
            df_ag = load_drug_autoencoded_AG(ncols=feature_subsample, scaling=scaling)
            df = df.merge(df_ag, on='NSC')
            input_dims['smiles_latent_AG'] = df_ag.shape[1] - 1
        elif fea == 'noise':
            df_drug_ids = df[['NSC']].drop_duplicates()
            noise = np.random.normal(size=(df_drug_ids.shape[0], 500))
            df_rand = pd.DataFrame(noise, index=df_drug_ids['NSC'],
                                   columns=['RAND-{:03d}'.format(x) for x in range(500)])
            df = df.merge(df_rand, on='NSC')
            input_dims['drug_noise'] = df_rand.shape[1] - 1

    df = df.set_index('NSC')

    if df.shape[0] and verbose:
        print('Loaded {} rows and {} columns'.format(df.shape[0], df.shape[1]))
        print('Input features:', ', '.join(['{}: {}'.format(k, v) for k, v in input_dims.items()]))

    return df


def load_by_drug_data(drug='1', cell_features=['expression'], shuffle=True,
                      use_gi50=False, logconc=-4., subsample='naive_balancing',
                      feature_subsample=None, scaling='std', scramble=False, verbose=True):

    """Load dataframe for by drug models

    Parameters
    ----------
    drug: drug NSC ID
    cell_features: list of strings from 'expression', 'expression_5platform', 'mirna', 'proteome', 'all' (default ['expression'])
        use one or more cell line feature sets: gene expression, microRNA, proteome
        use 'all' for ['expression', 'mirna', 'proteome']
    shuffle : True or False, optional (default True)
        if True shuffles the merged data before splitting training and validation sets
    scramble: True or False, optional (default False)
        if True randomly shuffle dose response data as a control
    use_gi50: True of False, optional (default False)
        use NCI GI50 value instead of percent growth at log concentration levels
    logconc: float value between -3 and -7, optional (default -4.)
        log concentration of drug to return cell line growth
    feature_subsample: None or integer (default None)
        number of feature columns to use from cellline expressions and drug descriptors
    scaling: None, 'std', 'minmax' or 'maxabs' (default 'std')
        type of feature scaling: 'maxabs' to [-1,1], 'maxabs' to [-1, 1], 'std' for standard normalization
    subsample: 'naive_balancing' or None
        if True balance dose response data with crude subsampling
    scramble: True or False, optional (default False)
        if True randomly shuffle dose response data as a control
    """

    if 'all' in cell_features:
        cell_features = ['expression', 'mirna', 'proteome']

    df_resp = load_dose_response(subsample=subsample, min_logconc=logconc, max_logconc=logconc, fraction=True)
    df_resp = df_resp.reset_index()

    df = df_resp[df_resp['NSC'] == drug]
    df = df[['CELLNAME', 'GROWTH']]

    input_dims = collections.OrderedDict()

    for fea in cell_features:
        if fea == 'expression' or fea == 'expression_u133p2':
            df_expr_u133p2 = load_cell_expression_u133p2(ncols=feature_subsample, scaling=scaling)
            df = df.merge(df_expr_u133p2, on='CELLNAME')
            input_dims['expression_u133p2'] = df_expr_u133p2.shape[1] - 1
        elif fea == 'expression_5platform':
            df_expr_5p = load_cell_expression_5platform(ncols=feature_subsample, scaling=scaling)
            df = df.merge(df_expr_5p, on='CELLNAME')
            input_dims['expression_5platform'] = df_expr_5p.shape[1] - 1
        elif fea == 'mirna':
            df_mirna = load_cell_mirna(ncols=feature_subsample, scaling=scaling)
            df = df.merge(df_mirna, on='CELLNAME')
            input_dims['microRNA'] = df_mirna.shape[1] - 1
        elif fea == 'proteome':
            df_prot = load_cell_proteome(ncols=feature_subsample, scaling=scaling)
            df = df.merge(df_prot, on='CELLNAME')
            input_dims['proteome'] = df_prot.shape[1] - 1

    df = df.set_index('CELLNAME')

    if df.shape[0] and verbose:
        print('Loaded {} rows and {} columns'.format(df.shape[0], df.shape[1]))
        print('Input features:', ', '.join(['{}: {}'.format(k, v) for k, v in input_dims.items()]))

    return df

#! /usr/bin/env python

from __future__ import division, print_function

import argparse
import collections
import logging
import os
import random
import threading

import numpy as np
import pandas as pd

from itertools import cycle, islice

import keras
from keras import backend as K
from keras import optimizers
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler, TensorBoard
from keras.utils import get_custom_objects
from keras.utils.vis_utils import plot_model
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
from scipy.stats.stats import pearsonr

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


import combo
import p1_common
# import p1_common_keras
from solr_keras import CandleRemoteMonitor, compute_trainable_params, TerminateOnTimeOut

# import argparser
# from datasets import NCI60

import NCI60
import combo


logger = logging.getLogger(__name__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(seed)

    random.seed(seed)

    if K.backend() == 'tensorflow':
        import tensorflow as tf
        tf.set_random_seed(seed)
        # session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
        # sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
        # K.set_session(sess)

        # Uncommit when running on an optimized tensorflow where NUM_INTER_THREADS and
        # NUM_INTRA_THREADS env vars are set.
        # session_conf = tf.ConfigProto(inter_op_parallelism_threads=int(os.environ['NUM_INTER_THREADS']),
        #	intra_op_parallelism_threads=int(os.environ['NUM_INTRA_THREADS']))
        # sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
        # K.set_session(sess)


def verify_path(path):
    folder = os.path.dirname(path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)


def set_up_logger(logfile, verbose):
    verify_path(logfile)
    fh = logging.FileHandler(logfile)
    fh.setFormatter(logging.Formatter("[%(asctime)s %(process)d] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    fh.setLevel(logging.DEBUG)

    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter(''))
    sh.setLevel(logging.DEBUG if verbose else logging.INFO)

    logger.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.addHandler(sh)


def extension_from_parameters(args):
    """Construct string for saving model with annotation of parameters"""
    ext = ''
    ext += '.A={}'.format(args.activation)
    ext += '.B={}'.format(args.batch_size)
    ext += '.E={}'.format(args.epochs)
    ext += '.O={}'.format(args.optimizer)
    # ext += '.LEN={}'.format(args.maxlen)
    ext += '.LR={}'.format(args.learning_rate)
    ext += '.CF={}'.format(''.join([x[0] for x in sorted(args.cell_features)]))
    ext += '.DF={}'.format(''.join([x[0] for x in sorted(args.drug_features)]))
    if args.feature_subsample > 0:
        ext += '.FS={}'.format(args.feature_subsample)
    if args.drop > 0:
        ext += '.DR={}'.format(args.drop)
    if args.warmup_lr:
        ext += '.wu_lr'
    if args.reduce_lr:
        ext += '.re_lr'
    if args.residual:
        ext += '.res'
    if args.use_landmark_genes:
        ext += '.L1000'
    if args.gen:
        ext += '.gen'
    if args.use_combo_score:
        ext += '.scr'
    for i, n in enumerate(args.dense):
        if n > 0:
            ext += '.D{}={}'.format(i+1, n)
    if args.dense_feature_layers != args.dense:
        for i, n in enumerate(args.dense):
            if n > 0:
                ext += '.FD{}={}'.format(i+1, n)

    return ext


def discretize(y, bins=5):
    percentiles = [100 / bins * (i + 1) for i in range(bins - 1)]
    thresholds = [np.percentile(y, x) for x in percentiles]
    classes = np.digitize(y, thresholds)
    return classes


class ComboDataLoader(object):
    """Load merged drug response, drug descriptors and cell line essay data
    """

    def __init__(self, seed, val_split=0.2, shuffle=True,
                 cell_features=['expression'], drug_features=['descriptors'],
                 use_landmark_genes=False, use_combo_score=False,
                 feature_subsample=None, scaling='std', scramble=False,
                 cv_partition='overlapping', cv=0):
        """Initialize data merging drug response, drug descriptors and cell line essay.
           Shuffle and split training and validation set

        Parameters
        ----------
        seed: integer
            seed for random generation
        val_split : float, optional (default 0.2)
            fraction of data to use in validation
        cell_features: list of strings from 'expression', 'expression_5platform', 'mirna', 'proteome', 'all', 'categorical' (default ['expression'])
            use one or more cell line feature sets: gene expression, microRNA, proteome
            use 'all' for ['expression', 'mirna', 'proteome']
            use 'categorical' for one-hot encoded cell lines
        drug_features: list of strings from 'descriptors', 'latent', 'all', 'categorical', 'noise' (default ['descriptors'])
            use dragon7 descriptors, latent representations from Aspuru-Guzik's SMILES autoencoder
            trained on NSC drugs, or both; use random features if set to noise
            use 'categorical' for one-hot encoded drugs
        shuffle : True or False, optional (default True)
            if True shuffles the merged data before splitting training and validation sets
        scramble: True or False, optional (default False)
            if True randomly shuffle dose response data as a control
        feature_subsample: None or integer (default None)
            number of feature columns to use from cellline expressions and drug descriptors
        use_landmark_genes: True or False
            only use LINCS1000 landmark genes
        use_combo_score: bool (default False)
            use combination score in place of percent growth (stored in 'GROWTH' column)
        scaling: None, 'std', 'minmax' or 'maxabs' (default 'std')
            type of feature scaling: 'maxabs' to [-1,1], 'maxabs' to [-1, 1], 'std' for standard normalization
        """

        self.cv_partition = cv_partition

        np.random.seed(seed)

        df = NCI60.load_combo_dose_response(use_combo_score=use_combo_score, fraction=True)
        logger.info('Loaded {} unique (CL, D1, D2) response sets.'.format(df.shape[0]))

        if 'all' in cell_features:
            self.cell_features = ['expression', 'mirna', 'proteome']
        else:
            self.cell_features = cell_features

        if 'all' in drug_features:
            self.drug_features = ['descriptors', 'latent']
        else:
            self.drug_features = drug_features

        for fea in self.cell_features:
            if fea == 'expression' or fea == 'rnaseq':
                self.df_cell_expr = NCI60.load_cell_expression_rnaseq(ncols=feature_subsample, scaling=scaling, use_landmark_genes=use_landmark_genes)
                df = df.merge(self.df_cell_expr[['CELLNAME']], on='CELLNAME')
            elif fea == 'expression_u133p2':
                self.df_cell_expr = NCI60.load_cell_expression_u133p2(ncols=feature_subsample, scaling=scaling, use_landmark_genes=use_landmark_genes)
                df = df.merge(self.df_cell_expr[['CELLNAME']], on='CELLNAME')
            elif fea == 'expression_5platform':
                self.df_cell_expr = NCI60.load_cell_expression_5platform(ncols=feature_subsample, scaling=scaling, use_landmark_genes=use_landmark_genes)
                df = df.merge(self.df_cell_expr[['CELLNAME']], on='CELLNAME')
            elif fea == 'mirna':
                self.df_cell_mirna = NCI60.load_cell_mirna(ncols=feature_subsample, scaling=scaling)
                df = df.merge(self.df_cell_mirna[['CELLNAME']], on='CELLNAME')
            elif fea == 'proteome':
                self.df_cell_prot = NCI60.load_cell_proteome(ncols=feature_subsample, scaling=scaling)
                df = df.merge(self.df_cell_prot[['CELLNAME']], on='CELLNAME')
            elif fea == 'categorical':
                df_cell_ids = df[['CELLNAME']].drop_duplicates()
                cell_ids = df_cell_ids['CELLNAME'].map(lambda x: x.replace(':', '.'))
                df_cell_cat = pd.get_dummies(cell_ids)
                df_cell_cat.index = df_cell_ids['CELLNAME']
                self.df_cell_cat = df_cell_cat.reset_index()

        for fea in self.drug_features:
            if fea == 'descriptors':
                self.df_drug_desc = NCI60.load_drug_descriptors(ncols=feature_subsample, scaling=scaling)
                df = df[df['NSC1'].isin(self.df_drug_desc['NSC']) & df['NSC2'].isin(self.df_drug_desc['NSC'])]
            elif fea == 'latent':
                self.df_drug_auen = NCI60.load_drug_autoencoded_AG(ncols=feature_subsample, scaling=scaling)
                df = df[df['NSC1'].isin(self.df_drug_auen['NSC']) & df['NSC2'].isin(self.df_drug_auen['NSC'])]
            elif fea == 'categorical':
                df_drug_ids = df[['NSC1']].drop_duplicates()
                df_drug_ids.columns = ['NSC']
                drug_ids = df_drug_ids['NSC']
                df_drug_cat = pd.get_dummies(drug_ids)
                df_drug_cat.index = df_drug_ids['NSC']
                self.df_drug_cat = df_drug_cat.reset_index()
            elif fea == 'noise':
                ids1 = df[['NSC1']].drop_duplicates().rename(columns={'NSC1':'NSC'})
                ids2 = df[['NSC2']].drop_duplicates().rename(columns={'NSC2':'NSC'})
                df_drug_ids = pd.concat([ids1, ids2]).drop_duplicates()
                noise = np.random.normal(size=(df_drug_ids.shape[0], 500))
                df_rand = pd.DataFrame(noise, index=df_drug_ids['NSC'],
                                       columns=['RAND-{:03d}'.format(x) for x in range(500)])
                self.df_drug_rand = df_rand.reset_index()

        logger.info('Filtered down to {} rows with matching information.'.format(df.shape[0]))

        ids1 = df[['NSC1']].drop_duplicates().rename(columns={'NSC1':'NSC'})
        ids2 = df[['NSC2']].drop_duplicates().rename(columns={'NSC2':'NSC'})
        df_drug_ids = pd.concat([ids1, ids2]).drop_duplicates().reset_index(drop=True)

        n_drugs = df_drug_ids.shape[0]
        n_val_drugs = int(n_drugs * val_split)
        n_train_drugs = n_drugs - n_val_drugs

        logger.info('Unique cell lines: {}'.format(df['CELLNAME'].nunique()))
        logger.info('Unique drugs: {}'.format(n_drugs))
        # df.to_csv('filtered.growth.min.tsv', sep='\t', index=False, float_format='%.4g')
        # df.to_csv('filtered.score.max.tsv', sep='\t', index=False, float_format='%.4g')

        if shuffle:
            df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
            df_drug_ids = df_drug_ids.sample(frac=1.0, random_state=seed).reset_index(drop=True)

        self.df_response = df
        self.df_drug_ids = df_drug_ids

        self.train_drug_ids = df_drug_ids['NSC'][:n_train_drugs]
        self.val_drug_ids = df_drug_ids['NSC'][-n_val_drugs:]

        if scramble:
            growth = df[['GROWTH']]
            random_growth = growth.iloc[np.random.permutation(np.arange(growth.shape[0]))].reset_index()
            self.df_response[['GROWTH']] = random_growth['GROWTH']
            logger.warn('Randomly shuffled dose response growth values.')

        logger.info('Distribution of dose response:')
        logger.info(self.df_response[['GROWTH']].describe())

        self.total = df.shape[0]
        self.n_val = int(self.total * val_split)
        self.n_train = self.total - self.n_val
        logger.info('Rows in train: {}, val: {}'.format(self.n_train, self.n_val))

        self.cell_df_dict = {'expression': 'df_cell_expr',
                             'expression_5platform': 'df_cell_expr',
                             'expression_u133p2': 'df_cell_expr',
                             'rnaseq': 'df_cell_expr',
                             'mirna': 'df_cell_mirna',
                             'proteome': 'df_cell_prot',
                             'categorical': 'df_cell_cat'}

        self.drug_df_dict = {'descriptors': 'df_drug_desc',
                             'latent': 'df_drug_auen',
                             'categorical': 'df_drug_cat',
                             'noise': 'df_drug_rand'}

        self.input_features = collections.OrderedDict()
        self.feature_shapes = {}
        for fea in self.cell_features:
            feature_type = 'cell.' + fea
            feature_name = 'cell.' + fea
            df_cell = getattr(self, self.cell_df_dict[fea])
            self.input_features[feature_name] = feature_type
            self.feature_shapes[feature_type] = (df_cell.shape[1] - 1,)

        for drug in ['drug1', 'drug2']:
            for fea in self.drug_features:
                feature_type = 'drug.' + fea
                feature_name = drug + '.' + fea
                df_drug = getattr(self, self.drug_df_dict[fea])
                self.input_features[feature_name] = feature_type
                self.feature_shapes[feature_type] = (df_drug.shape[1] - 1,)

        self.feature_shapes['dose'] = (1,)
        for dose in ['dose1', 'dose2']:
            self.input_features[dose] = 'dose'

        logger.info('Input features shapes:')
        for k, v in self.input_features.items():
            logger.info('  {}: {}'.format(k, self.feature_shapes[v]))

        self.input_dim = sum([np.prod(self.feature_shapes[x]) for x in self.input_features.values()])
        logger.info('Total input dimensions: {}'.format(self.input_dim))

        if cv > 1:
            if cv_partition == 'disjoint':
                pass
            elif cv_partition == 'disjoint_cells':
                y = self.df_response['GROWTH'].values
                groups = self.df_response['CELLNAME'].values
                gkf = GroupKFold(n_splits=cv)
                splits = gkf.split(y, groups=groups)
                self.cv_train_indexes = []
                self.cv_val_indexes = []
                for index, (train_index, val_index) in enumerate(splits):
                    print(index, train_index)
                    self.cv_train_indexes.append(train_index)
                    self.cv_val_indexes.append(val_index)
            else:
                y = self.df_response['GROWTH'].values
                # kf = KFold(n_splits=cv)
                # splits = kf.split(y)
                skf = StratifiedKFold(n_splits=cv, random_state=seed)
                splits = skf.split(y, discretize(y, bins=cv))
                self.cv_train_indexes = []
                self.cv_val_indexes = []
                for index, (train_index, val_index) in enumerate(splits):
                    print(index, train_index)
                    self.cv_train_indexes.append(train_index)
                    self.cv_val_indexes.append(val_index)

    def load_data_all(self, switch_drugs=False):
        df_all = self.df_response
        y_all = df_all['GROWTH'].values
        x_all_list = []

        for fea in self.cell_features:
            df_cell = getattr(self, self.cell_df_dict[fea])
            df_x_all = pd.merge(df_all[['CELLNAME']], df_cell, on='CELLNAME', how='left')
            x_all_list.append(df_x_all.drop(['CELLNAME'], axis=1).values)

        # for fea in loader.cell_features:
        #     df_cell = getattr(loader, loader.cell_df_dict[fea])
        #     df_x_all = pd.merge(df_all[['CELLNAME']], df_cell, on='CELLNAME', how='left')
        #     df_x_all[:1000].to_csv('df.{}.1k.csv'.format(fea), index=False, float_format="%g")

        drugs = ['NSC1', 'NSC2']
        doses = ['pCONC1', 'pCONC2']
        if switch_drugs:
            drugs = ['NSC2', 'NSC1']
            doses = ['pCONC2', 'pCONC1']

        for drug in drugs:
            for fea in self.drug_features:
                df_drug = getattr(self, self.drug_df_dict[fea])
                df_x_all = pd.merge(df_all[[drug]], df_drug, left_on=drug, right_on='NSC', how='left')
                x_all_list.append(df_x_all.drop([drug, 'NSC'], axis=1).values)

        for dose in doses:
            x_all_list.append(df_all[dose].values)

        # for drug in drugs:
        #     for fea in loader.drug_features:
        #         df_drug = getattr(loader, loader.drug_df_dict[fea])
        #         df_x_all = pd.merge(df_all[[drug]], df_drug, left_on=drug, right_on='NSC', how='left')
        #         print(df_x_all.shape)
        #         df_x_all[:1000].drop([drug], axis=1).to_csv('df.{}.{}.1k.csv'.format(drug, fea), index=False, float_format="%g")

        # df_all[:1000].to_csv('df.growth.1k.csv', index=False, float_format="%g")

        return x_all_list, y_all, df_all

    def load_data_by_index(self, train_index, val_index):
        x_all_list, y_all, df_all = self.load_data_all()
        x_train_list = [x[train_index] for x in x_all_list]
        x_val_list = [x[val_index] for x in x_all_list]
        y_train = y_all[train_index]
        y_val = y_all[val_index]
        df_train = df_all.iloc[train_index, :]
        df_val = df_all.iloc[val_index, :]
        if self.cv_partition == 'disjoint':
            logger.info('Training drugs: {}'.format(set(df_train['NSC1'])))
            logger.info('Validation drugs: {}'.format(set(df_val['NSC1'])))
        elif self.cv_partition == 'disjoint_cells':
            logger.info('Training cells: {}'.format(set(df_train['CELLNAME'])))
            logger.info('Validation cells: {}'.format(set(df_val['CELLNAME'])))
        return x_train_list, y_train, x_val_list, y_val, df_train, df_val

    def load_data_cv(self, fold):
        train_index = self.cv_train_indexes[fold]
        val_index = self.cv_val_indexes[fold]
        # print('fold', fold)
        # print(train_index[:5])
        return self.load_data_by_index(train_index, val_index)

    def load_data(self):
        if self.cv_partition == 'disjoint':
            train_index = self.df_response[(self.df_response['NSC1'].isin(self.train_drug_ids)) & (self.df_response['NSC2'].isin(self.train_drug_ids))].index
            val_index = self.df_response[(self.df_response['NSC1'].isin(self.val_drug_ids)) & (self.df_response['NSC2'].isin(self.val_drug_ids))].index
        else:
            train_index = range(self.n_train)
            val_index = range(self.n_train, self.total)
        return self.load_data_by_index(train_index, val_index)

    def load_data_old(self):
        # bad performance (4x slow) possibly due to incontiguous data
        df_train = self.df_response.iloc[:self.n_train, :]
        df_val = self.df_response.iloc[self.n_train:, :]

        y_train = df_train['GROWTH'].values
        y_val = df_val['GROWTH'].values

        x_train_list = []
        x_val_list = []

        for fea in self.cell_features:
            df_cell = getattr(self, self.cell_df_dict[fea])
            df_x_train = pd.merge(df_train[['CELLNAME']], df_cell, on='CELLNAME', how='left')
            df_x_val = pd.merge(df_val[['CELLNAME']], df_cell, on='CELLNAME', how='left')
            x_train_list.append(df_x_train.drop(['CELLNAME'], axis=1).values)
            x_val_list.append(df_x_val.drop(['CELLNAME'], axis=1).values)

        for drug in ['NSC1', 'NSC2']:
            for fea in self.drug_features:
                df_drug = getattr(self, self.drug_df_dict[fea])
                df_x_train = pd.merge(df_train[[drug]], df_drug, left_on=drug, right_on='NSC', how='left')
                df_x_val = pd.merge(df_val[[drug]], df_drug, left_on=drug, right_on='NSC', how='left')
                x_train_list.append(df_x_train.drop([drug, 'NSC'], axis=1).values)
                x_val_list.append(df_x_val.drop([drug, 'NSC'], axis=1).values)

        return x_train_list, y_train, x_val_list, y_val, df_train, df_val


class ComboDataGenerator(object):
    """Generate training, validation or testing batches from loaded data
    """
    def __init__(self, data, partition='train', batch_size=32):
        self.lock = threading.Lock()
        self.data = data
        self.partition = partition
        self.batch_size = batch_size

        if partition == 'train':
            self.cycle = cycle(range(data.n_train))
            self.num_data = data.n_train
        elif partition == 'val':
            self.cycle = cycle(range(data.total)[-data.n_val:])
            self.num_data = data.n_val
        else:
            raise Exception('Data partition "{}" not recognized.'.format(partition))

    def flow(self):
        """Keep generating data batches
        """
        while 1:
            self.lock.acquire()
            indices = list(islice(self.cycle, self.batch_size))
            self.lock.release()

            df = self.data.df_response.iloc[indices, :]
            y = df['GROWTH'].values

            x_list = []

            for fea in self.data.cell_features:
                df_cell = getattr(self.data, self.data.cell_df_dict[fea])
                df_x = pd.merge(df[['CELLNAME']], df_cell, on='CELLNAME', how='left')
                x_list.append(df_x.drop(['CELLNAME'], axis=1).values)

            for drug in ['NSC1', 'NSC2']:
                for fea in self.data.drug_features:
                    df_drug = getattr(self.data, self.data.drug_df_dict[fea])
                    df_x = pd.merge(df[[drug]], df_drug, left_on=drug, right_on='NSC', how='left')
                    x_list.append(df_x.drop([drug, 'NSC'], axis=1).values)

            yield x_list, y


def test_generator(loader):
    gen = ComboDataGenerator(loader).flow()
    x_list, y = next(gen)
    for x in x_list:
        print(x.shape)
    print(y.shape)


def test_loader(loader):
    x_train_list, y_train, x_val_list, y_val = loader.load_data()
    print('x_train shapes:')
    for x in x_train_list:
        print(x.shape)
    print('y_train shape:', y_train.shape)

    print('x_val shapes:')
    for x in x_val_list:
        print(x.shape)
    print('y_val shape:', y_val.shape)


def r2(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res/(SS_tot + K.epsilon()))


def mae(y_true, y_pred):
    return keras.metrics.mean_absolute_error(y_true, y_pred)


def evaluate_prediction(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    corr, _ = pearsonr(y_true, y_pred)
    return {'mse': mse, 'mae': mae, 'r2': r2, 'corr': corr}


def log_evaluation(metric_outputs, description='Comparing y_true and y_pred:'):
    logger.info(description)
    for metric, value in metric_outputs.items():
        logger.info('  {}: {:.4f}'.format(metric, value))


def plot_history(out, history, metric='loss', title=None):
    title = title or 'model {}'.format(metric)
    val_metric = 'val_{}'.format(metric)
    plt.figure(figsize=(8, 6))
    plt.plot(history.history[metric], marker='o')
    plt.plot(history.history[val_metric], marker='d')
    plt.title(title)
    plt.ylabel(metric)
    plt.xlabel('epoch')
    plt.legend(['train_{}'.format(metric), 'val_{}'.format(metric)], loc='upper center')
    png = '{}.plot.{}.png'.format(out, metric)
    plt.savefig(png, bbox_inches='tight')


class LoggingCallback(Callback):
    def __init__(self, print_fcn=print):
        Callback.__init__(self)
        self.print_fcn = print_fcn

    def on_epoch_end(self, epoch, logs={}):
        msg = "[Epoch: %i] %s" % (epoch, ", ".join("%s: %f" % (k, v) for k, v in sorted(logs.items())))
        self.print_fcn(msg)


class PermanentDropout(Dropout):
    def __init__(self, rate, **kwargs):
        super(PermanentDropout, self).__init__(rate, **kwargs)
        self.uses_learning_phase = False

    def call(self, x, mask=None):
        if 0. < self.rate < 1.:
            noise_shape = self._get_noise_shape(x)
            x = K.dropout(x, self.rate, noise_shape)
        return x


class ModelRecorder(Callback):
    def __init__(self, save_all_models=False):
        Callback.__init__(self)
        self.save_all_models = save_all_models
        get_custom_objects()['PermanentDropout'] = PermanentDropout

    def on_train_begin(self, logs={}):
        self.val_losses = []
        self.best_val_loss = np.Inf
        self.best_model = None

    def on_epoch_end(self, epoch, logs={}):
        val_loss = logs.get('val_loss')
        self.val_losses.append(val_loss)
        if val_loss < self.best_val_loss:
            self.best_model = keras.models.clone_model(self.model)
            self.best_val_loss = val_loss


def build_feature_model(input_shape, name='', dense_layers=[1000, 1000],
                        activation='relu', residual=False,
                        dropout_rate=0, permanent_dropout=True):
    x_input = Input(shape=input_shape)
    h = x_input
    for i, layer in enumerate(dense_layers):
        x = h
        h = Dense(layer, activation=activation)(h)
        if dropout_rate > 0:
            if permanent_dropout:
                h = PermanentDropout(dropout_rate)(h)
            else:
                h = Dropout(dropout_rate)(h)
        if residual:
            try:
                h = keras.layers.add([h, x])
            except ValueError:
                pass
    model = Model(x_input, h, name=name)
    return model


def build_model(loader, args, verbose=False):
    input_models = {}
    dropout_rate = args.drop
    permanent_dropout = True
    for fea_type, shape in loader.feature_shapes.items():
        box = build_feature_model(input_shape=shape, name=fea_type,
                                  dense_layers=args.dense_feature_layers,
                                  dropout_rate=dropout_rate, permanent_dropout=permanent_dropout)
        if verbose:
            box.summary()
        input_models[fea_type] = box

    inputs = []
    encoded_inputs = []
    for fea_name, fea_type in loader.input_features.items():
        shape = loader.feature_shapes[fea_type]
        fea_input = Input(shape, name='input.'+fea_name)
        inputs.append(fea_input)
        input_model = input_models[fea_type]
        encoded = input_model(fea_input)
        encoded_inputs.append(encoded)

    merged = keras.layers.concatenate(encoded_inputs)

    h = merged
    for i, layer in enumerate(args.dense):
        x = h
        h = Dense(layer, activation=args.activation)(h)
        if dropout_rate > 0:
            if permanent_dropout:
                h = PermanentDropout(dropout_rate)(h)
            else:
                h = Dropout(dropout_rate)(h)
        if args.residual:
            try:
                h = keras.layers.add([h, x])
            except ValueError:
                pass
    output = Dense(1)(h)

    return Model(inputs, output)



def get_combo_parser():
    description = 'Build neural network based models to predict tumor response to drug pairs.'
    parser = argparse.ArgumentParser(prog='combo_baseline', formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description=description)
    return combo.common_parser(parser)


def initialize_parameters():
    # Get command-line parameters
    parser = get_combo_parser()
    args = parser.parse_args()
    # Get parameters from configuration file
    file_params = combo.read_config_file(args.config_file)
    # Consolidate parameter set. Command-line parameters overwrite file configuration
    params = p1_common.args_overwrite_config(args, file_params)
    # print(params)
    return params


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def run(params):
    args = Struct(**params)
    set_seed(args.rng_seed)
    ext = extension_from_parameters(args)
    prefix = args.save + ext
    logfile = args.logfile if args.logfile else prefix+'.log'
    set_up_logger(logfile, args.verbose)
    logger.info('Params: {}'.format(params))

    loader = ComboDataLoader(seed=args.rng_seed,
                             val_split=args.validation_split,
                             cell_features=args.cell_features,
                             drug_features=args.drug_features,
                             use_landmark_genes=args.use_landmark_genes,
                             use_combo_score=args.use_combo_score,
                             cv_partition=args.cv_partition, cv=args.cv)
    # test_loader(loader)
    # test_generator(loader)

    train_gen = ComboDataGenerator(loader, batch_size=args.batch_size).flow()
    val_gen = ComboDataGenerator(loader, partition='val', batch_size=args.batch_size).flow()

    train_steps = int(loader.n_train / args.batch_size)
    val_steps = int(loader.n_val / args.batch_size)

    model = build_model(loader, args, verbose=True)
    model.summary()
    # plot_model(model, to_file=prefix+'.model.png', show_shapes=True)

    if args.cp:
        model_json = model.to_json()
        with open(prefix+'.model.json', 'w') as f:
            print(model_json, file=f)

    def warmup_scheduler(epoch):
        lr = args.learning_rate or base_lr * args.batch_size/100
        if epoch <= 5:
            K.set_value(model.optimizer.lr, (base_lr * (5-epoch) + lr * epoch) / 5)
        logger.debug('Epoch {}: lr={}'.format(epoch, K.get_value(model.optimizer.lr)))
        return K.get_value(model.optimizer.lr)

    df_pred_list = []

    cv_ext = ''
    cv = args.cv if args.cv > 1 else 1

    fold = 0
    while fold < cv:
        if args.cv > 1:
            logger.info('Cross validation fold {}/{}:'.format(fold+1, cv))
            cv_ext = '.cv{}'.format(fold+1)

        model = build_model(loader, args)

        optimizer = optimizers.deserialize({'class_name': args.optimizer, 'config': {}})
        base_lr = args.base_lr or K.get_value(optimizer.lr)
        if args.learning_rate:
            K.set_value(optimizer.lr, args.learning_rate)

        model.compile(loss=args.loss, optimizer=optimizer, metrics=[mae, r2])

        # calculate trainable and non-trainable params
        params.update(compute_trainable_params(model))

        candle_monitor = CandleRemoteMonitor(params=params)
        timeout_monitor = TerminateOnTimeOut(params['timeout'])

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)
        warmup_lr = LearningRateScheduler(warmup_scheduler)
        checkpointer = ModelCheckpoint(prefix+cv_ext+'.weights.h5', save_best_only=True, save_weights_only=True)
        tensorboard = TensorBoard(log_dir="tb/tb{}{}".format(ext, cv_ext))
        history_logger = LoggingCallback(logger.debug)
        model_recorder = ModelRecorder()

        # callbacks = [history_logger, model_recorder]
        callbacks = [candle_monitor, timeout_monitor, history_logger, model_recorder]
        if args.reduce_lr:
            callbacks.append(reduce_lr)
        if args.warmup_lr:
            callbacks.append(warmup_lr)
        if args.cp:
            callbacks.append(checkpointer)
        if args.tb:
            callbacks.append(tensorboard)

        if args.gen:
            history = model.fit_generator(train_gen, train_steps,
                                          epochs=args.epochs,
                                          callbacks=callbacks,
                                          validation_data=val_gen, validation_steps=val_steps)
        else:
            if args.cv > 1:
                x_train_list, y_train, x_val_list, y_val, df_train, df_val = loader.load_data_cv(fold)
            else:
                x_train_list, y_train, x_val_list, y_val, df_train, df_val = loader.load_data()

            y_shuf = np.random.permutation(y_val)
            log_evaluation(evaluate_prediction(y_val, y_shuf),
                           description='Between random pairs in y_val:')
            history = model.fit(x_train_list, y_train,
                                batch_size=args.batch_size,
                                shuffle=args.shuffle,
                                epochs=args.epochs,
                                callbacks=callbacks,
                                validation_data=(x_val_list, y_val))

        if args.cp:
            model.load_weights(prefix+cv_ext+'.weights.h5')

        if not args.gen:
            y_val_pred = model.predict(x_val_list, batch_size=args.batch_size).flatten()
            scores = evaluate_prediction(y_val, y_val_pred)
            if args.cv > 1 and scores[args.loss] > args.max_val_loss:
                logger.warn('Best val_loss {} is greater than {}; retrain the model...'.format(scores[args.loss], args.max_val_loss))
                continue
            else:
                fold += 1
            log_evaluation(scores)
            df_val.is_copy = False
            df_val['GROWTH_PRED'] = y_val_pred
            df_val['GROWTH_ERROR'] = y_val_pred - y_val
            df_pred_list.append(df_val)

        if args.cp:
            # model.save(prefix+'.model.h5')
            model_recorder.best_model.save(prefix+'.model.h5')

            # test reloadded model prediction
            new_model = keras.models.load_model(prefix+'.model.h5')
            new_model.load_weights(prefix+cv_ext+'.weights.h5')
            new_pred = new_model.predict(x_val_list, batch_size=args.batch_size).flatten()
            # print('y_val:', y_val[:10])
            # print('old_pred:', y_val_pred[:10])
            # print('new_pred:', new_pred[:10])

        plot_history(prefix, history, 'loss')
        plot_history(prefix, history, 'r2')

        if K.backend() == 'tensorflow':
            K.clear_session()

    pred_fname = prefix + '.predicted.growth.tsv'
    if args.use_combo_score:
        pred_fname = prefix + '.predicted.score.tsv'
    df_pred = pd.concat(df_pred_list)
    df_pred.to_csv(pred_fname, sep='\t', index=False, float_format='%.4g')

    logger.handlers = []

    return history


def main():
    params = initialize_parameters()
    run(params)


if __name__ == '__main__':
    main()
    if K.backend() == 'tensorflow':
        K.clear_session()

#! /usr/bin/env python

from __future__ import division, print_function

import argparse
import os

import numpy as np
import pandas as pd
import keras
from keras import backend as K
from keras.models import Model
from keras.utils import get_custom_objects
from tqdm import tqdm

import NCI60


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if K.backend() == 'tensorflow' and 'NUM_INTRA_THREADS' in os.environ:
    import tensorflow as tf
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)

    # Uncommit when running on an optimized tensorflow where NUM_INTER_THREADS and
    # NUM_INTRA_THREADS env vars are set.
    print('NUM_INTER_THREADS: ', os.environ['NUM_INTER_THREADS'])
    print('NUM_INTRA_THREADS: ', os.environ['NUM_INTRA_THREADS'])
    session_conf = tf.ConfigProto(inter_op_parallelism_threads=int(os.environ['NUM_INTER_THREADS']),
          intra_op_parallelism_threads=int(os.environ['NUM_INTRA_THREADS']))
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)


class PermanentDropout(keras.layers.Dropout):
    def __init__(self, rate, **kwargs):
        super(PermanentDropout, self).__init__(rate, **kwargs)
        self.uses_learning_phase = False

    def call(self, x, mask=None):
        if 0. < self.rate < 1.:
            noise_shape = self._get_noise_shape(x)
            x = K.dropout(x, self.rate, noise_shape)
        return x


def get_parser(description=None):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-s', '--sample_set',
                        default='NCIPDM',
                        help='cell sample set: NCI60, NCIPDM, GDSC, ...')
    parser.add_argument('-d', '--drug_set',
                        default='ALMANAC',
                        help='drug set: ALMANAC, GDSC, NCI_IOA_AOA, ...')
    parser.add_argument('-z', '--batch_size', type=int,
                        default=100000,
                        help='batch size')
    parser.add_argument('--step', type=int,
                        default=10000,
                        help='number of rows to inter in each step')
    parser.add_argument('-m', '--model_file',
                        default='saved.model.h5',
                        help='trained model file')
    parser.add_argument('-n', '--n_pred', type=int,
                        default=1,
                        help='the number of predictions to make for each sample-drug combination for uncertainty quantification')
    parser.add_argument('-w', '--weights_file',
                        default='saved.weights.h5',
                        help='trained weights file (loading model file alone sometimes does not work in keras)')
    parser.add_argument('--ns', type=int,
                        default=0,
                        help='the first n entries of cell samples to subsample')
    parser.add_argument('--nd', type=int,
                        default=0,
                        help='the first n entries of drugs to subsample')
    parser.add_argument("--si", type=int,
                         default=0,
                         help='the index of the first cell sample to subsample')
    parser.add_argument("--use_landmark_genes", action="store_true",
                        help="use the 978 landmark genes from LINCS (L1000) as expression features")

    return parser


def lookup(df, sample, drug1, drug2=None, value=None):
    drug2 = drug2 or drug1
    df_result = df[(df['Sample'] == sample) & (df['Drug1'] == drug1) & (df['Drug2'] == drug2)]
    if df_result.empty:
        df_result = df[(df['Sample'] == sample) & (df['Drug1'] == drug2) & (df['Drug2'] == drug1)]
    if value:
        if df_result.empty:
            return 1.0
        else:
            return df_result[value].iloc[0]
    else:
        return df_result


def custom_combo_score(combined_growth, growth_1, growth_2):
    if growth_1 <= 0 or growth_2 <= 0:
        expected_growth = min(growth_1, growth_2)
    else:
        expected_growth = growth_1 * growth_2
    custom_score = (expected_growth - combined_growth) * 100
    return custom_score


def cross_join(df1, df2, **kwargs):
    df1['_tmpkey'] = 1
    df2['_tmpkey'] = 1

    res = pd.merge(df1, df2, on='_tmpkey', **kwargs).drop('_tmpkey', axis=1)
    # res.index = pd.MultiIndex.from_product((df1.index, df2.index))

    df1.drop('_tmpkey', axis=1, inplace=True)
    df2.drop('_tmpkey', axis=1, inplace=True)

    return res


def cross_join3(df1, df2, df3, **kwargs):
    return cross_join(cross_join(df1, df2), df3, **kwargs)


def prepare_data(sample_set='NCI60', drug_set='ALMANAC', use_landmark_genes=False):
    df_expr = NCI60.load_sample_rnaseq(use_landmark_genes=use_landmark_genes, sample_set=sample_set)
    # df_old = NCI60.load_cell_expression_rnaseq(use_landmark_genes=True)
    # df_desc = NCI60.load_drug_descriptors_new()
    df_desc = NCI60.load_drug_set_descriptors(drug_set=drug_set)
    return df_expr, df_desc

def initialize_parameters():
    description = 'Infer drug pair response from trained combo model.'
    parser = get_parser(description)
    args = parser.parse_args()
    return args

def run():
    args = initialize_parameters()

    get_custom_objects()['PermanentDropout'] = PermanentDropout
    model = keras.models.load_model(args.model_file, compile=False)
    model.load_weights(args.weights_file)
    # model.summary()

    df_expr, df_desc = prepare_data(sample_set=args.sample_set, drug_set=args.drug_set, use_landmark_genes=args.use_landmark_genes)

    print('total available samples: ', df_expr[['Sample']].shape[0])
    print('total available drugs: ', df_desc[['Drug']].shape[0])

    if args.ns > 0 and args.si > 0:
        df_sample_ids = df_expr[['Sample']].iloc[args.si:args.si+args.ns]
    elif args.si > 0:
        df_sample_ids = df_expr[['Sample']].iloc[args.si:]
    elif args.ns > 0:
        df_sample_ids = df_expr[['Sample']].head(args.ns)
    else:
        df_sample_ids = df_expr[['Sample']].copy()
    if args.nd > 0:
        df_drug_ids = df_desc[['Drug']].head(args.nd)
    else:
        df_drug_ids = df_desc[['Drug']].copy()

    df_sum = cross_join3(df_sample_ids, df_drug_ids, df_drug_ids, suffixes=('1', '2'))

    n_samples = df_sample_ids.shape[0]
    n_drugs = df_drug_ids.shape[0]
    n_rows = n_samples * n_drugs * n_drugs

    print('Predicting drug response for {} combinations: {} samples x {} drugs x {} drugs'.format(n_rows, n_samples, n_drugs, n_drugs))
    n = args.n_pred
    df_sum['N'] = n
    df_seq = pd.DataFrame({'Seq': range(1, n+1)})
    df_all = cross_join(df_sum, df_seq)

    total = df_sum.shape[0]
    for i in tqdm(range(0, total, args.step)):
        j = min(i+args.step, total)

        x_all_list = []
        df_x_all = pd.merge(df_all[['Sample']].iloc[i:j], df_expr, on='Sample', how='left')
        x_all_list.append(df_x_all.drop(['Sample'], axis=1).values)

        drugs = ['Drug1', 'Drug2']
        for drug in drugs:
            df_x_all = pd.merge(df_all[[drug]].iloc[i:j], df_desc, left_on=drug, right_on='Drug', how='left')
            x_all_list.append(df_x_all.drop([drug, 'Drug'], axis=1).values)

        preds = []
        for k in range(n):
            y_pred = model.predict(x_all_list, batch_size=args.batch_size, verbose=0).flatten()
            preds.append(y_pred)
            df_all.loc[i*n+k:(j-1)*n+k:n, 'PredGrowth'] = y_pred
            df_all.loc[i*n+k:(j-1)*n+k:n, 'Seq'] = k + 1

        if n > 0:
            df_sum.loc[i:j-1, 'PredGrowthMean'] = np.mean(preds, axis=0)
            df_sum.loc[i:j-1, 'PredGrowthStd'] = np.std(preds, axis=0)
            df_sum.loc[i:j-1, 'PredGrowthMin'] = np.min(preds, axis=0)
            df_sum.loc[i:j-1, 'PredGrowthMax'] = np.max(preds, axis=0)

    # df = df_all.copy()
    # df['PredCustomComboScore'] = df.apply(lambda x: custom_combo_score(x['PredGrowth'],
    #                                                                    lookup(df, x['Sample'], x['Drug1'], value='PredGrowth'),
    #                                                                    lookup(df, x['Sample'], x['Drug2'], value='PredGrowth')), axis=1)

    csv_all = 'comb_pred_{}_{}.all.tsv'.format(args.sample_set, args.drug_set)
    df_all.to_csv(csv_all, index=False, sep='\t', float_format='%.4f')

    if n > 0:
        csv = 'comb_pred_{}_{}.tsv'.format(args.sample_set, args.drug_set)
        df_sum.to_csv(csv, index=False, sep='\t', float_format='%.4f')


if __name__ == '__main__':
    run()
    if K.backend() == 'tensorflow':
        K.clear_session()

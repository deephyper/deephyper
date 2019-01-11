from __future__ import print_function
from __future__ import division

import operator
import os
import re
import warnings
import numpy as np

from sklearn import metrics
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import *
from sklearn.ensemble import *
from sklearn.neighbors import *
from sklearn.svm import *

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from xgboost import XGBRegressor
    from xgboost import XGBClassifier


def get_model(model_or_name, threads=-1, classification=False):
    regression_models = {
        'xgboost': (XGBRegressor(max_depth=6, nthread=threads), 'XGBRegressor'),
        'randomforest': (RandomForestRegressor(n_estimators=100, n_jobs=threads), 'RandomForestRegressor'),
        'adaboost': (AdaBoostRegressor(), 'AdaBoostRegressor'),
        'linear': (LinearRegression(), 'LinearRegression'),
        'elasticnet': (ElasticNetCV(positive=True), 'ElasticNetCV'),
        'lasso': (LassoCV(positive=True), 'LassoCV'),
        'ridge': (Ridge(), 'Ridge'),

        'xgb.1k': (XGBRegressor(max_depth=6, n_estimators=1000, nthread=threads), 'XGBRegressor.1K'),
        'xgb.10k': (XGBRegressor(max_depth=6, n_estimators=10000, nthread=threads), 'XGBRegressor.10K'),
        'rf.1k': (RandomForestRegressor(n_estimators=1000, n_jobs=threads), 'RandomForestRegressor.1K'),
        'rf.10k': (RandomForestRegressor(n_estimators=10000, n_jobs=threads), 'RandomForestRegressor.10K')
    }

    classification_models = {
        'xgboost': (XGBClassifier(nthread=threads), 'XGBClassifier'),
        'randomforest': (RandomForestClassifier(n_estimators=100, n_jobs=threads), 'RandomForestClassifier'),
        'adaboost': (AdaBoostClassifier(), 'AdaBoostClassifier'),
        'logistic': (LogisticRegression(), 'LogisticRegression'),
        'gaussian': (GaussianProcessClassifier(), 'GaussianProcessClassifier'),
        'knn': (KNeighborsClassifier(), 'KNeighborsClassifier'),
        'bayes': (GaussianNB(), 'GaussianNB'),
        'svm': (SVC(), 'SVC'),

        'xgb.1k': (XGBClassifier(n_estimators=1000, nthread=threads), 'XGBClassifier.1K'),
        'rf.1k': (RandomForestClassifier(n_estimators=1000, n_jobs=threads), 'RandomForestClassifier.1K'),
        'xgb.10k': (XGBClassifier(n_estimators=10000, nthread=threads), 'XGBClassifier.10K'),
        'rf.10k': (RandomForestClassifier(n_estimators=10000, n_jobs=threads), 'RandomForestClassifier.10K')
    }

    if isinstance(model_or_name, str):
        if classification:
            model_and_name = classification_models.get(model_or_name.lower())
        else:
            model_and_name = regression_models.get(model_or_name.lower())
        if not model_and_name:
            raise Exception("unrecognized model: '{}'".format(model_or_name))
        else:
            model, name = model_and_name
    else:
        model = model_or_name
        name = re.search("\w+", str(model)).group(0)

    return model, name


def score_format(metric, score, signed=False, eol=''):
    if signed:
        return '{:<25} = {:+.5f}'.format(metric, score) + eol
    else:
        return '{:<25} =  {:.5f}'.format(metric, score) + eol


def top_important_features(model, feature_names, n_top=1000):
    if hasattr(model, "booster"): # XGB
        fscore = model.booster().get_fscore()
        fscore = sorted(fscore.items(), key=operator.itemgetter(1), reverse=True)
        features = [(v, feature_names[int(k[1:])]) for k,v in fscore]
        top = features[:n_top]
    else:
        if hasattr(model, "feature_importances_"):
            fi = model.feature_importances_
        else:
            if hasattr(model, "coef_"):
                fi = model.coef_
            else:
                return
        features = [(f, n) for f, n in zip(fi, feature_names)]
        top = sorted(features, key=lambda f:abs(f[0]), reverse=True)[:n_top]
    return top


def sprint_features(top_features, n_top=1000):
    str = ''
    for i, feature in enumerate(top_features):
        if i >= n_top:
            break
        str += '{:9.5f}\t{}\n'.format(feature[0], feature[1])
    return str


def discretize(y, bins=5, cutoffs=None, min_count=0, verbose=False):
    thresholds = cutoffs
    if thresholds is None:
        percentiles = [100 / bins * (i + 1) for i in range(bins - 1)]
        thresholds = [np.percentile(y, x) for x in percentiles]
    classes = np.digitize(y, thresholds)
    good_bins = None
    if verbose:
        bc = np.bincount(classes)
        good_bins = len(bc)
        min_y = np.min(y)
        max_y = np.max(y)
        print('Category cutoffs:', ['{:.3g}'.format(t) for t in thresholds])
        print('Bin counts:')
        for i, count in enumerate(bc):
            lower = min_y if i == 0 else thresholds[i-1]
            upper = max_y if i == len(bc)-1 else thresholds[i]
            removed = ''
            if count < min_count:
                removed = ' .. removed (<{})'.format(min_count)
                good_bins -= 1
            print('  Class {}: {:7d} ({:.4f}) - between {:+.2f} and {:+.2f}{}'.
                  format(i, count, count/len(y), lower, upper, removed))
        # print('  Total: {:9d}'.format(len(y)))
    return classes, thresholds, good_bins


def categorize_dataframe(df, bins=5, cutoffs=None, verbose=False):
    y = df.as_matrix()[:, 0]
    classes, _, _ = discretize(y, bins, cutoffs, verbose)
    df.iloc[:, 0] = classes
    return df


def summarize(df, cutoffs=None, autobins=0, min_count=0):
    mat = df.as_matrix()
    x, y = mat[:, 1:], mat[:, 0]
    y_discrete, thresholds, _ = discretize(y, bins=4)
    print('Quartiles of y:', ['{:.2g}'.format(t) for t in thresholds])
    good_bins = None
    if cutoffs or autobins > 1:
        _, _, good_bins = discretize(y, bins=autobins, cutoffs=cutoffs, min_count=min_count, verbose=True)
    print()
    return good_bins


def regress(model, data, cv=5, cutoffs=None, threads=-1, prefix=''):
    out_dir = os.path.dirname(prefix)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    model, name = get_model(model, threads)
    mat = data.as_matrix()
    x, y = mat[:, 1:], mat[:, 0]
    feature_labels = data.columns.tolist()[1:]

    train_scores, test_scores = [], []
    tests, preds = None, None
    best_model = None
    best_score = -np.Inf

    y_even, _, _ = discretize(y)

    print('>', name)
    print('Cross validation:')
    skf = StratifiedKFold(n_splits=cv, shuffle=True)
    for i, (train_index, test_index) in enumerate(skf.split(x, y_even)):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(x_train, y_train)
        train_score = model.score(x_train, y_train)
        test_score = model.score(x_test, y_test)
        train_scores.append(train_score)
        test_scores.append(test_score)
        print("  fold {}/{}: score = {:.3f}".format(i+1, cv, test_score))
        if test_score > best_score:
            best_model = model
        y_pred = model.predict(x_test)
        preds = np.concatenate((preds, y_pred)) if preds is not None else y_pred
        tests = np.concatenate((tests, y_test)) if tests is not None else y_test

    print('Average validation metrics:')
    scores_fname = "{}.{}.scores".format(prefix, name)
    metric_names = 'r2_score explained_variance_score mean_absolute_error mean_squared_error'.split()
    with open(scores_fname, "w") as scores_file:
        for m in metric_names:
            try:
                s = getattr(metrics, m)(tests, preds)
                print(' ', score_format(m, s))
                scores_file.write(score_format(m, s, eol='\n'))
            except Exception:
                pass
        scores_file.write('\nModel:\n{}\n\n'.format(model))

    print()
    top_features = top_important_features(best_model, feature_labels)
    if top_features is not None:
        fea_fname = "{}.{}.features".format(prefix, name)
        with open(fea_fname, "w") as fea_file:
            fea_file.write(sprint_features(top_features))


def classify(model, data, cv=5, cutoffs=None, autobins=0, threads=-1, prefix=''):
    out_dir = os.path.dirname(prefix)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    model, name = get_model(model, threads, classification=True)
    mat = data.as_matrix()
    x, y = mat[:, 1:], mat[:, 0]
    feature_labels = data.columns.tolist()[1:]

    if autobins >= 2:
        y, cutoffs, _ = discretize(y, bins=autobins, min_count=cv)
    elif cutoffs:
        y, _, _ = discretize(y, cutoffs=cutoffs)

    mask = np.ones(len(y), dtype=bool)
    bc = np.bincount(y)
    for i, count in enumerate(bc):
        if count < cv:
            mask[y == i] = False
    x = x[mask]
    y = y[mask]

    train_scores, test_scores = [], []
    tests, preds = None, None
    probas = None
    best_model = None
    best_score = -np.Inf

    print('>', name)
    print('Cross validation:')
    skf = StratifiedKFold(n_splits=cv, shuffle=True)
    for i, (train_index, test_index) in enumerate(skf.split(x, y)):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(x_train, y_train)
        train_score = model.score(x_train, y_train)
        test_score = model.score(x_test, y_test)
        train_scores.append(train_score)
        test_scores.append(test_score)
        print("  fold {}/{}: score = {:.3f}".format(i+1, cv, test_score))
        if test_score > best_score:
            best_model = model
        y_pred = model.predict(x_test)
        preds = np.concatenate((preds, y_pred)) if preds is not None else y_pred
        tests = np.concatenate((tests, y_test)) if tests is not None else y_test
        if hasattr(model, "predict_proba"):
            probas_ = model.predict_proba(x_test)
            probas = np.concatenate((probas, probas_)) if probas is not None else probas_

    roc_auc_score = None
    if probas is not None:
        fpr, tpr, thresholds = metrics.roc_curve(tests, probas[:, 1], pos_label=0)
        roc_auc_score = metrics.auc(fpr, tpr)
        roc_fname = "{}.{}.ROC".format(prefix, name)
        if roc_auc_score:
            with open(roc_fname, "w") as roc_file:
                roc_file.write('\t'.join(['Threshold', 'FPR', 'TPR'])+'\n')
                for ent in zip(thresholds, fpr, tpr):
                    roc_file.write('\t'.join("{0:.5f}".format(x) for x in list(ent))+'\n')

    print('Average validation metrics:')
    naive_accuracy = max(np.bincount(tests)) / len(tests)
    accuracy = np.sum(preds == tests) / len(tests)
    accuracy_gain = accuracy - naive_accuracy
    print(' ', score_format('accuracy_gain', accuracy_gain, signed=True))
    scores_fname = "{}.{}.scores".format(prefix, name)
    metric_names = 'accuracy_score f1_score precision_score recall_score log_loss'.split()
    with open(scores_fname, "w") as scores_file:
        scores_file.write(score_format('accuracy_gain', accuracy_gain, signed=True, eol='\n'))
        for m in metric_names:
            try:
                s = getattr(metrics, m)(tests, preds)
                print(' ', score_format(m, s))
                scores_file.write(score_format(m, s, eol='\n'))
            except Exception:
                pass
        if roc_auc_score:
            print(' ', score_format('roc_auc_score', roc_auc_score))
            scores_file.write(score_format('roc_auc_score', roc_auc_score, eol='\n'))
        scores_file.write('\nModel:\n{}\n\n'.format(model))

    print()
    top_features = top_important_features(best_model, feature_labels)
    if top_features is not None:
        fea_fname = "{}.{}.features".format(prefix, name)
        with open(fea_fname, "w") as fea_file:
            fea_file.write(sprint_features(top_features))

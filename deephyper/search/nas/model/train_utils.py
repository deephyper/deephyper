from collections import OrderedDict

import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error

loss_metrics = OrderedDict()
loss_metrics['mean_absolute_error'] = lambda x,y: tf.reduce_mean(tf.abs(x-y))
loss_metrics['mean_squared_error'] = tf.losses.mean_squared_error
loss_metrics['sigmoid_cross_entropy'] = tf.losses.sigmoid_cross_entropy
loss_metrics['mean_softmax_cross_entropy'] = lambda la, lo: tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=lo, labels=la))
loss_metrics['sequence_loss_by_example'] = 'sequence_loss_by_example'

test_metrics = OrderedDict()
test_metrics['mean_absolute_error'] = mean_absolute_error
test_metrics['mae'] = test_metrics['mean_absolute_error']
test_metrics['mean_squared_error'] = mean_squared_error
test_metrics['mse'] = test_metrics['mean_squared_error']
test_metrics['accuracy'] = lambda preds, labels: 100.0 * np.sum(np.argmax(preds, 1) == np.argmax(labels, 1)) / preds.shape[0]
test_metrics['acc'] = test_metrics['accuracy']
test_metrics['perplexity'] = 'perplexity'

optimizers = OrderedDict()
optimizers['sgd']     = tf.train.GradientDescentOptimizer
optimizers['rmsprop'] = tf.train.RMSPropOptimizer
optimizers['adagrad'] = tf.train.AdagradOptimizer
optimizers['adam']    = tf.train.AdamOptimizer
optimizers['momentum'] = lambda x : tf.train.MomentumOptimizer(x, momentum=0.9)

optimizers_keras = OrderedDict()
optimizers_keras['sgd']     = tf.keras.optimizers.SGD
optimizers_keras['rmsprop'] = tf.keras.optimizers.RMSprop
optimizers_keras['adagrad'] = tf.keras.optimizers.Adagrad
optimizers_keras['adam']    = tf.keras.optimizers.Adam

def selectTestMetric(name):
    '''
      Return the test_metric defined by name.
    '''
    if (test_metrics.get(name) == None):
        raise RuntimeError('"{0}" is not a defined test_metric.'.format(name))
    else:
        return test_metrics[name]


def selectLossMetric(name):
    '''
      Return the loss_metric defined by name.
    '''
    if (loss_metrics.get(name) == None):
        raise RuntimeError('"{0}" is not a defined loss_metric.'.format(name))
    else:
        return loss_metrics[name]

def selectOptimizer(name):
    '''
      Return the optimizer defined by name.
    '''
    if (optimizers.get(name) == None):
        raise RuntimeError('"{0}" is not a defined optimizer.'.format(name))
    else:
        return optimizers[name]

def selectOptimizer_keras(name):
    '''
      Return the optimizer defined by name.
    '''
    if (optimizers_keras.get(name) == None):
        raise RuntimeError('"{0}" is not a defined optimizer for keras.'.format(name))
    else:
        return optimizers_keras[name]

def check_data_config(data_dict):
    gen_keys = ["train_gen", "train_size", "valid_gen", "valid_size", "types", "shapes"]
    ndarray_keys = ["train_X", "train_Y", "valid_X", "valid_Y"]
    if all([k in data_dict.keys() for k in gen_keys]):
        return "gen"
    elif all([k in data_dict.keys() for k in ndarray_keys]):
        return "ndarray"
    else:
        raise RuntimeError('Wrong data config...')


# Metrics with tensors

def r2(y_true, y_pred):
    SS_res =  tf.keras.backend.sum(tf.keras.backend.square(y_true - y_pred))
    SS_tot = tf.keras.backend.sum(tf.keras.backend.square(y_true - tf.keras.backend.mean(y_true)))
    return (1 - SS_res/(SS_tot + tf.keras.backend.epsilon()))

def mae(y_true, y_pred):
    return tf.keras.metrics.mean_absolute_error(y_true, y_pred)

def mse(y_true, y_pred):
    return tf.keras.metrics.mean_squared_error(y_true, y_pred)

def acc(y_true, y_pred):
    return tf.keras.metrics.categorical_accuracy(y_true, y_pred)

metrics = OrderedDict()
metrics['mean_absolute_error'] = metrics['mae'] = mae
metrics['r2'] = r2

def selectMetric(name):
    '''
      Return the metric defined by name.
    '''
    if (metrics.get(name) == None):
        return name # supposing it is referenced in keras metrics
    else:
        return metrics[name]




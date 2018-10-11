'''
 * @Author: romain.egele
 * @Date: 2018-06-19 16:14:04
 * @Last Modified by:   romain.egele
 * @Last Modified time: 2018-06-19 16:14:04
'''
import tensorflow as tf
import numpy as np
from collections import OrderedDict
from sklearn.metrics import mean_absolute_error, mean_squared_error


loss_metrics = OrderedDict()
loss_metrics['mean_absolute_error'] = lambda x,y: tf.reduce_mean(tf.abs(x-y))
loss_metrics['mean_squared_error'] = tf.losses.mean_squared_error
loss_metrics['sigmoid_cross_entropy'] = tf.losses.sigmoid_cross_entropy
loss_metrics['softmax_cross_entropy'] = tf.losses.softmax_cross_entropy
loss_metrics['softmax_cross_entropy_v2'] = lambda x,y: tf.nn.softmax_cross_entropy_with_logits_v2(labels=x, logits=y)
loss_metrics['sequence_loss_by_example'] = 'sequence_loss_by_example'
test_metrics = OrderedDict()
test_metrics['mean_absolute_error'] = mean_absolute_error
test_metrics['mean_squared_error'] = mean_squared_error
test_metrics['accuracy'] = lambda preds, labels: 100.0 * np.sum(np.argmax(preds, 1) == np.argmax(labels)) / preds.shape[0]
test_metrics['perplexity'] = 'perplexity'

optimizers = OrderedDict()
optimizers['sgd']     = tf.train.GradientDescentOptimizer
optimizers['rmsprop'] = tf.train.RMSPropOptimizer
optimizers['adagrad'] = tf.train.AdagradOptimizer
optimizers['adam']    = tf.train.AdamOptimizer
optimizers['momentum'] = lambda x : tf.train.MomentumOptimizer(x, momentum=0.9)

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

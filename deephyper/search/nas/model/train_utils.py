from collections import OrderedDict

import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error

from ... import util

optimizers_keras = OrderedDict()
optimizers_keras["sgd"] = tf.keras.optimizers.SGD
optimizers_keras["rmsprop"] = tf.keras.optimizers.RMSprop
optimizers_keras["adagrad"] = tf.keras.optimizers.Adagrad
optimizers_keras["adam"] = tf.keras.optimizers.Adam
optimizers_keras["adadelta"] = tf.keras.optimizers.Adadelta
optimizers_keras["adamax"] = tf.keras.optimizers.Adamax
optimizers_keras["nadam"] = tf.keras.optimizers.Nadam


def selectOptimizer_keras(name):
    """Return the optimizer defined by name.
    """
    if optimizers_keras.get(name) == None:
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
        raise RuntimeError("Wrong data config...")


# Metrics with tensors


# def r2(y_true, y_pred):
#     SS_res = tf.keras.backend.sum(tf.keras.backend.square(y_true - y_pred))
#     SS_tot = tf.keras.backend.sum(
#         tf.keras.backend.square(y_true - tf.keras.backend.mean(y_true))
#     )
#     return 1 - SS_res / (SS_tot + tf.keras.backend.epsilon())


def r2(y_true, y_pred):
    SS_res = tf.keras.backend.sum(tf.keras.backend.square(y_true - y_pred), axis=0)
    SS_tot = tf.keras.backend.sum(
        tf.keras.backend.square(y_true - tf.keras.backend.mean(y_true, axis=0)), axis=0
    )
    output_scores = 1 - SS_res / (SS_tot + tf.keras.backend.epsilon())
    r2 = tf.keras.backend.mean(output_scores)
    return r2


def mae(y_true, y_pred):
    return tf.keras.metrics.mean_absolute_error(y_true, y_pred)


def mse(y_true, y_pred):
    return tf.keras.metrics.mean_squared_error(y_true, y_pred)


def acc(y_true, y_pred):
    return tf.keras.metrics.categorical_accuracy(y_true, y_pred)


metrics = OrderedDict()
metrics["mean_absolute_error"] = metrics["mae"] = mae
metrics["r2"] = r2
metrics["mean_squared_error"] = metrics["mse"] = mse
metrics["accuracy"] = metrics["acc"] = acc


def selectMetric(name: str):
    """Return the metric defined by name.

    Args:
        name (str): a string referenced in DeepHyper, one referenced in keras or an attribute name to import.

    Returns:
        str or callable: a string suppossing it is referenced in the keras framework or a callable taking (y_true, y_pred) as inputs and returning a tensor.
    """
    if metrics.get(name) == None:
        try:
            return util.load_attr_from(name)
        except:
            return name  # supposing it is referenced in keras metrics
    else:
        return metrics[name]

from collections import OrderedDict

import tensorflow as tf
import tensorflow.keras.backend as K

from deephyper.search import util

optimizers_keras = OrderedDict()
optimizers_keras["sgd"] = tf.keras.optimizers.SGD
optimizers_keras["rmsprop"] = tf.keras.optimizers.RMSprop
optimizers_keras["adagrad"] = tf.keras.optimizers.Adagrad
optimizers_keras["adam"] = tf.keras.optimizers.Adam
optimizers_keras["adadelta"] = tf.keras.optimizers.Adadelta
optimizers_keras["adamax"] = tf.keras.optimizers.Adamax
optimizers_keras["nadam"] = tf.keras.optimizers.Nadam


def selectOptimizer_keras(name):
    """Return the optimizer defined by name."""
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


def negmae(y_true, y_pred):
    return -mae(y_true, y_pred)


def mse(y_true, y_pred):
    return tf.keras.metrics.mean_squared_error(y_true, y_pred)


def negmse(y_true, y_pred):
    return -mse(y_true, y_pred)


def acc(y_true, y_pred):
    return tf.keras.metrics.categorical_accuracy(y_true, y_pred)


def sparse_perplexity(y_true, y_pred):
    cross_entropy = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    perplexity = tf.pow(2.0, cross_entropy)
    return perplexity

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_score(y_true, y_pred):
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


metrics = OrderedDict()
metrics["mean_absolute_error"] = metrics["mae"] = mae
metrics["negative_mean_absolute_error"] = metrics["negmae"] = negmae
metrics["r2"] = r2
metrics["mean_squared_error"] = metrics["mse"] = mse
metrics["negative_mean_squared_error"] = metrics["negmse"] = negmse
metrics["accuracy"] = metrics["acc"] = acc
metrics["sparse_perplexity"] = sparse_perplexity
metrics["f1_score"] = f1_score


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

"""This module provides different metric functions. A metric can be defined by a keyword (str) or a callable. If it is a keyword it has to be available in ``tensorflow.keras`` or in ``deephyper.netrics``. The loss functions availble in ``deephyper.metrics`` are:
* Sparse Perplexity: ``sparse_perplexity``
* R2: ``r2``
* AUC ROC: ``auroc``
* AUC Precision-Recall: ``aucpr``
"""
import functools
from collections import OrderedDict

import tensorflow as tf
from deephyper.core.utils import load_attr


def r2(y_true, y_pred):
    SS_res = tf.math.reduce_sum(tf.math.square(y_true - y_pred), axis=0)
    SS_tot = tf.math.reduce_sum(
        tf.math.square(y_true - tf.math.reduce_mean(y_true, axis=0)), axis=0
    )
    output_scores = 1 - SS_res / (SS_tot + tf.keras.backend.epsilon())
    r2 = tf.math.reduce_mean(output_scores)
    return r2


def mae(y_true, y_pred):
    return tf.keras.metrics.mean_absolute_error(y_true, y_pred)


def mse(y_true, y_pred):
    return tf.keras.metrics.mean_squared_error(y_true, y_pred)


def rmse(y_true, y_pred):
    return tf.math.sqrt(tf.math.reduce_mean(tf.math.square(y_pred - y_true)))


def acc(y_true, y_pred):
    return tf.keras.metrics.categorical_accuracy(y_true, y_pred)


def sparse_perplexity(y_true, y_pred):
    cross_entropy = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    perplexity = tf.pow(2.0, cross_entropy)
    return perplexity


def to_tfp(metric_func):
    """Convert a regular tensorflow-keras metric for tensorflow probability where the output is a distribution.

    Args:
        metric_func (func): A regular tensorflow-keras metric function.
    """

    @functools.wraps(metric_func)
    def wrapper(y_true, y_pred):
        return metric_func(y_true, y_pred.mean())

    wrapper.__name__ = f"tfp_{metric_func.__name__}"

    return wrapper


# convert some metrics for Tensorflow Probability where the output of the model is
# a distribution
tfp_r2 = to_tfp(r2)
tfp_mae = to_tfp(mae)
tfp_mse = to_tfp(mse)
tfp_rmse = to_tfp(rmse)

metrics_func = OrderedDict()
metrics_func["mean_absolute_error"] = metrics_func["mae"] = mae
metrics_func["r2"] = r2
metrics_func["mean_squared_error"] = metrics_func["mse"] = mse
metrics_func["root_mean_squared_error"] = metrics_func["rmse"] = rmse
metrics_func["accuracy"] = metrics_func["acc"] = acc
metrics_func["sparse_perplexity"] = sparse_perplexity

metrics_func["tfp_r2"] = tfp_r2
metrics_func["tfp_mse"] = tfp_mse
metrics_func["tfp_mae"] = tfp_mae
metrics_func["tfp_rmse"] = tfp_rmse

metrics_obj = OrderedDict()
metrics_obj["auroc"] = lambda: tf.keras.metrics.AUC(name="auroc", curve="ROC")
metrics_obj["aucpr"] = lambda: tf.keras.metrics.AUC(name="aucpr", curve="PR")


def selectMetric(name: str):
    """Return the metric defined by name.

    Args:
        name (str): a string referenced in DeepHyper, one referenced in keras or an attribute name to import.

    Returns:
        str or callable: a string suppossing it is referenced in the keras framework or a callable taking (y_true, y_pred) as inputs and returning a tensor.
    """
    if callable(name):
        return name
    if metrics_func.get(name) is None and metrics_obj.get(name) is None:
        try:
            return load_attr(name)
        except Exception:
            return name  # supposing it is referenced in keras metrics
    else:
        if name in metrics_func:
            return metrics_func[name]
        else:
            return metrics_obj[name]()

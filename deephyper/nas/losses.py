"""This module provides different loss functions. A loss can be defined by a keyword (str) or a callable following the ``tensorflow.keras`` interface. If it is a keyword it has to be available in ``tensorflow.keras`` or in ``deephyper.losses``. The loss functions availble in ``deephyper.losses`` are:
* Negative Log Likelihood (compatible with Tensorflow Probability): ``tfp_negloglik`` or ``tfp_nll``
"""
from collections import OrderedDict

import tensorflow as tf
from deephyper.core.utils import load_attr


def tfp_negloglik(y, rv_y):
    """Negative log likelihood for Tensorflow probability."""
    return -rv_y.log_prob(y)


losses_func = OrderedDict()
losses_func["tfp_negloglik"] = losses_func["tfp_nll"] = tfp_negloglik

losses_obj = OrderedDict()


def selectLoss(name: str):
    """Return the loss defined by name.

    Args:
        name (str): a string referenced in DeepHyper, one referenced in keras or an attribute name to import.

    Returns:
        str or callable: a string suppossing it is referenced in the keras framework or a callable taking (y_true, y_pred) as inputs and returning a tensor.
    """
    if callable(name):
        return name
    if losses_func.get(name) is None and losses_obj.get(name) is None:
        try:
            loaded_obj = load_attr(name)
            return loaded_obj
        except Exception:
            return tf.keras.losses.get(
                name
            )  # supposing it is referenced in keras losses
    else:
        if name in losses_func:
            return losses_func[name]
        else:
            return losses_obj[name]()

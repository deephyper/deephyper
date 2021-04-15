from collections import OrderedDict
import traceback

import tensorflow as tf
from deephyper.search import util

def tfp_negloglik(y, rv_y):
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
    if losses_func.get(name) == None and losses_obj.get(name) == None:
        try:
            loaded_obj = util.load_attr_from(name)
            return loaded_obj
        except:
            return name  # supposing it is referenced in keras losses
    else:
        if name in losses_func:
            return losses_func[name]
        else:
            return losses_obj[name]()

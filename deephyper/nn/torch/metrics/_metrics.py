from collections import OrderedDict

from deephyper.core.utils import load_attr

metrics_func = OrderedDict()

metrics_obj = OrderedDict()


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

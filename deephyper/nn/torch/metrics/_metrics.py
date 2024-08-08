from collections import OrderedDict

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
    elif name in metrics_func:
        return metrics_func[name]
    elif name in metrics_obj:
        return metrics_obj[name]()
    else:
        raise ValueError(f"Metric '{name}' not found!")

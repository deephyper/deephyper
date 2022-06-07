import importlib


def load_attr(str_full_module):
    """Loadd attribute from module.

    Args:
        str_full_module (str): string of the form ``{module_name}.{attr}``.

    Returns:
        Any: the attribute.
    """
    if type(str_full_module) == str:
        split_full = str_full_module.split(".")
        str_module = ".".join(split_full[:-1])
        str_attr = split_full[-1]
        module = importlib.import_module(str_module)
        return getattr(module, str_attr)
    else:
        return str_full_module

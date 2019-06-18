"""Exceptions related with imports of modules/attributes/scripts.
"""
from deephyper.core.exceptions import DeephyperError


class GenericLoaderError(DeephyperError):
    """Raised when the generic_loader function is failing.
    """

    def __init__(self, str_value):
        self.str_value = str_value

    def __str__(self):
        return f"The target '{self.str_value}' cannot be imported because it's neither a python script nor a python module."

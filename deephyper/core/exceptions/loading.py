"""Exceptions related with imports of modules/attributes/scripts.
"""
from deephyper.core.exceptions import DeephyperError


class GenericLoaderError(DeephyperError):
    """Raised when the generic_loader function is failing."""

    def __init__(self, target, attr, error_source, custom_msg=""):
        self.target = target
        self.attr = attr
        self.error_source = error_source
        self.custom_msg = custom_msg

    def __str__(self):
        error = (
            f"{self.error_source}\n"
            f"{self.custom_msg}"
            f"The attribute '{self.attr}' cannot be importe from '{self.target}'."
        )
        return error

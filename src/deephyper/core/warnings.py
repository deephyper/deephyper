"""Submodule containing code related to warnings."""

import warnings

# Used to enforce the warnings to be displayed
warnings.simplefilter("default")


def deprecated_api(msg: str) -> None:
    """Utility function to throw a deprecation warning for some section of the API."""
    warnings.warn(msg, DeprecationWarning, stacklevel=2)

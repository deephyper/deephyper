import warnings

# Used to enforce the warnings to be displayed
warnings.simplefilter("default")


def deprecated_api(msg: str) -> None:
    warnings.warn(msg, DeprecationWarning, stacklevel=2)

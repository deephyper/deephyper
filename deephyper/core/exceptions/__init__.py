"""Deephyper exceptions
"""

# ! Root exceptions


class DeephyperError(Exception):
    """Root deephyper exception."""


class DeephyperRuntimeError(RuntimeError):
    """Raised when an error is detected in deephyper and that doesn’t fall in any of the other categories. The associated value is a string indicating what precisely went wrong."""


class SearchTerminationError(RuntimeError):
    """Raised when a search receives SIGALARM"""


class RunFunctionError(RuntimeError):
    """Raised when error occurs in run-function"""

    def __init__(self, msg: str = None) -> None:
        self.msg = msg

    def __str__(self) -> str:
        return self.msg


class MissingRequirementError(RuntimeError):
    """Raised when a requirement is not installed properly."""

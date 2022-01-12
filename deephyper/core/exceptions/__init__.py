"""Deephyper exceptions
"""

# ! Root exceptions


class DeephyperError(Exception):
    """Root deephyper exception."""


class DeephyperRuntimeError(RuntimeError):
    """Raised when an error is detected in deephyper and that doesn’t fall in any of the other categories. The associated value is a string indicating what precisely went wrong."""


class SearchTerminationError(RuntimeError):
    """Raised when a search receives SIGALARM"""

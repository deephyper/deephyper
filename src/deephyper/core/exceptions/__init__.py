"""Set of root exceptions for the package."""


class DeephyperError(Exception):
    """Root deephyper exception."""


class DeephyperRuntimeError(RuntimeError):
    """Raised when for error that doesn't fall in other categories.

    The associated value is a string indicating what precisely went wrong.
    """


class SearchTerminationError(RuntimeError):
    """Raised when a search is terminated."""


class MaximumJobsSpawnReached(SearchTerminationError):
    """Raised when the maximum number of jobs is reached."""


class TimeoutReached(SearchTerminationError):
    """Raised when the timeout of the search was reached."""


class RunFunctionError(RuntimeError):
    """Raised when error occurs in run-function."""

    def __init__(self, msg: str = None) -> None:
        self.msg = msg

    def __str__(self) -> str:  # noqa: D105
        return self.msg


class MissingRequirementError(RuntimeError):
    """Raised when a requirement is not installed properly."""

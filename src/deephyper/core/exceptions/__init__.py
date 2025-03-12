"""Set of root exceptions for the package."""

class SearchTerminationError(RuntimeError):
    """Raised when a search is terminated."""


class MaximumJobsSpawnReached(SearchTerminationError):
    """Raised when the maximum number of jobs is reached."""


class TimeoutReached(SearchTerminationError):
    """Raised when the timeout of the search was reached."""


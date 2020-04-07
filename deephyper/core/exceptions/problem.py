"""Exceptions related with problem definition.
"""

from deephyper.core.exceptions import DeephyperError


class SpaceDimNameOfWrongType(DeephyperError):
    """Raised when a dimension name of the space is not a string."""

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return f"Dimension name: '{self.value}' is of type == {type(self.value)} when should be 'str'!"


# ! NaProblemErrors


class NaProblemError(DeephyperError):
    """Raise when an error occurs in a NaProblem instance."""

    def __init__(self, msg: str):
        self.msg = msg

    def __str__(self):
        return self.msg


class SearchSpaceBuilderIsNotCallable(NaProblemError):
    """Raised when a search space builder is not a callable."""

    def __init__(self, parameter):
        self.parameter = parameter

    def __str__(self):
        raise f"The search space builder {self.parameter} should be a callable when it is not!"


class SearchSpaceBuilderMissingParameter(NaProblemError):
    """Raised when a missing parameter is detected in a callable which creates a Structure.

        Args:
            missing_parameter (str): name of the missing parameter.
    """

    def __init__(self, missing_parameter):
        self.missing_parameter = missing_parameter

    def __str__(self):
        return f"The callable which creates a Structure is missing a '{self.missing_parameter}' parameter!"


class SearchSpaceBuilderMissingDefaultParameter(NaProblemError):
    """Raised when a parameter of a search space builder is missing a default value."""

    def __init__(self, parameter):
        self.parameter = parameter

    def __str__(self):
        return f"The parameter {self.parameter} must have a default value!"


class ProblemPreprocessingIsNotCallable(NaProblemError):
    """Raised when the preprocessing parameter is not callable."""

    def __init__(self, parameter):
        self.parameter = parameter

    def __str__(self):
        return f"The parameter {self.parameter} must be a callable."


class ProblemLoadDataIsNotCallable(NaProblemError):
    """Raised when the load_data parameter is not callable."""

    def __init__(self, parameter):
        self.parameter = parameter

    def __str__(self):
        return f"The parameter {self.parameter} must be a callable."


class WrongProblemObjective(NaProblemError):
    """Raised when the objective parameter is neither a callable nor a string."""

    def __init__(self, objective, possible_names=None):
        self.objective = objective
        self.possible_names = possible_names

    def __str__(self):
        output = f"The objective: {str(self.objective)} is not valid."
        if self.possible_names is not None:
            output += f" Possible objectives are: {self.possible_names}"
        return output

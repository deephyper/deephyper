"""Exceptions related with problem definition.
"""

from deephyper.core.exceptions import DeephyperError


class SpaceDimNameMismatch(DeephyperError):
    """"Raised when 2 set of keys are not corresponding for a given Problem.
    """

    def __init__(self, ref, space):
        self.ref, self.space = ref, space

    def __str__(self):
        return f'Some reference\'s dimensions doesn\'t exist in this space: {filter(lambda k: k in self.space, self.ref.keys())}'


class SpaceNumDimMismatch(DeephyperError):
    """Raised when 2 set of keys doesn't have the same number of keys for a given
    Problem."""

    def __init__(self, ref, space):
        self.ref, self.space = ref, space

    def __str__(self):
        return f'The reference has {len(self.ref)} dimensions when the space has {len(self.space)}. Both should have the same number of dimensions.'


class SpaceDimNameOfWrongType(DeephyperError):
    """Raised when a dimension name of the space is not a string."""

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return f"Dimension name: '{self.value}' is of type == {type(self.value)} when should be 'str'!"


class SpaceDimValueOfWrongType(DeephyperError):
    """Raised when a dimension value of the space is not a string."""

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return f"Dimension value: '{self.value}' is of type == {type(self.value)} when should be either 'tuple' or 'list'!"


class SpaceDimValueNotInSpace(DeephyperError):
    """Raised when a dimension value of the space is in the coresponding dimension's space."""

    def __init__(self, value, name_dim, space_dim):
        self.value = value
        self.name_dim = name_dim
        self.space_dim = space_dim

    def __str__(self):
        return f"Dimension value: '{self.value}' is not in dim['{self.name_dim}':{self.space_dim}!"

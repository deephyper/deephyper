from collections import OrderedDict
from pprint import pformat

from deephyper.core.exceptions.problem import *


class Problem:
    """Representation of a problem.

    Attribute:
        space (OrderedDict): represents the search space of the problem.
    """

    def __init__(self):
        self.__space = OrderedDict()

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return f'Problem\n{pformat({k:v for k,v in self.__space.items()}, indent=2)}'

    def add_dim(self, p_name, p_space):
        """Add a dimension to the search space.

        Args:
            p_name (str): name of the parameter/dimension.
            p_space (Object): space corresponding to the new dimension.
        """
        self.__space[p_name] = p_space

    @property
    def space(self):
        dims = list(self.__space.keys())
        dims.sort()
        space = OrderedDict(**{d: self.__space[d] for d in dims})
        return space


class HpProblem(Problem):
    """Problem specification for Hyperparameter Optimization"""

    def __init__(self):
        super().__init__()
        # * starting points
        self.references = []

    def __repr__(self):
        prob = super().__repr__()
        start = f'{pformat({k:v for k,v in enumerate(self.starting_point_asdict)})}'
        return prob + '\n\nStarting Point\n' + start

    def add_dim(self, p_name, p_space):
        """Add a dimension to the search space.

        Args:
            p_name (str): name of the parameter/dimension.
            p_space (tuple(int, int) or tuple(float, float) or list(Object,)): space corresponding to the new dimension.
        """
        if not type(p_name) is str:
            raise SpaceDimNameOfWrongType(p_name)

        if not type(p_space) is tuple \
                and not type(p_space) is list:
            raise SpaceDimValueOfWrongType(p_space)

        super().add_dim(p_name, p_space)

    def add_starting_point(self, **dims):
        """Add a new starting point to the problem.

        Args:
            **dims:

        Raises:
            SpaceNumDimMismatch: Raised when 2 set of keys doesn't have the same number of keys for a given Problem.
            SpaceDimNameMismatch: Raised when 2 set of keys are not corresponding for a given Problem.
            SpaceDimValueNotInSpace: Raised when a dimension value of the space is in the coresponding dimension's space.
        """

        if len(dims) != len(self.space):
            raise SpaceNumDimMismatch(dims, self.space)

        if not all(d in self.space for d in dims):
            raise SpaceDimNameMismatch(dims, self.space)

        for dim, value in zip(dims, dims.values()):
            if type(self.space[dim]) is list:
                if not value in self.space[dim]:
                    raise SpaceDimValueNotInSpace(value, dim, self.space[dim])
            else:  # * type(self.space[dim]) is tuple
                if value < self.space[dim][0] \
                        or value > self.space[dim][1]:
                    raise SpaceDimValueNotInSpace(value, dim, self.space[dim])

        self.references.append([dims[d] for d in self.space])

    @property
    def starting_point(self):
        """Starting point(s) of the search space.

        Returns:
            list(list): list of starting points where each point is a list of values. Values are indexed in the same order as the order of creation of space's dimensions.
        """
        return self.references

    @property
    def starting_point_asdict(self):
        """Starting point(s) of the search space.

        Returns:
            list(dict): list of starting points where each point is a dict of values. Each key are correspnding to dimensions of the space.
        """
        return [{k: v for k, v in zip(list(self.space.keys()), p)} for p in self.references]

from pprint import pformat
from collections import OrderedDict

# ! Exceptions


class SpaceDimNameMismatch(RuntimeError):
    """"When 2 set of keys are not corresponding for a given Problem.
    """

    def __init__(self, ref, space):
        self.ref, self.space = ref, space

    def __str__(self):
        return f'Some reference\'s dimensions doesn\'t exist in this space: {filter(lambda k: k in self.space, self.ref.keys())}'


class SpaceNumDimMismatch(RuntimeError):
    """When 2 set of keys doesn't have the same number of keys for a given
    Problem."""

    def __init__(self, ref, space):
        self.ref, self.space = ref, space

    def __str__(self):
        return f'The reference has {len(self.ref)} dimensions when the space has {len(self.space)}. Both should have the same number of dimensions.'


class SpaceDimNameOfWrongType(Exception):
    """When a dimension name of the space is not a string."""

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return f"Dimension name: '{self.value}' is of type == {type(self.value)} when should be 'str'!"


class SpaceDimValueOfWrongType(Exception):
    """When a dimension value of the space is not a string."""

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return f"Dimension value: '{self.value}' is of type == {type(self.value)} when should be either 'tuple' or 'list'!"

# ! Problems


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
            p_space (Object): space corresponding to the new dimension.
        """
        if not type(p_name) is str:
            raise SpaceDimNameOfWrongType(p_name)

        if not type(p_space) is tuple \
                and not type(p_space) is list:
            raise SpaceDimValueOfWrongType(p_space)

        super().add_dim(p_name, p_space)

    def add_reference(self, **dims):
        """Add a new starting point to the problem.

        Raises:
            RuntimeError: if one key of the 'dims' argument doesn't exist in the space.
        """
        space = self.space

        if len(dims) != len(space):
            raise SpaceNumDimMismatch(dims, space)

        if not all(d in space for d in dims):
            raise SpaceDimNameMismatch(dims, space)

        self.references.append([dims[d] for d in space])

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

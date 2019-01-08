from pprint import pformat
from collections import OrderedDict

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
        return self.__space.copy()

class HpProblem(Problem):
    """Problem specification for Hyperparameter Optimization"""

    def __init__(self):
        super().__init__()
        self.__def_values = OrderedDict()

    def __repr__(self):
        prob = super().__repr__()
        start = f'{pformat({k:v for k,v in self.starting_point_asdict.items()})}'
        return prob + '\n\nStarting Point\n' + start

    def add_dim(self, p_name, p_space, default=None):
        """Add a dimension to the search space.

        Args:
            p_name (str): name of the parameter/dimension.
            p_space (Object): space corresponding to the new dimension.
            default: default value of the new dimension, it must be compatible with the ``p_space`` given.
        """
        assert type(p_name) is str, f'p_name must be str type, got {type(p_name)} !'
        assert type(p_space) is tuple or type(p_space) is list, f'p_space must be tuple or list type, got {type(p_space)} !'
        super().add_dim(p_name, p_space)
        self.__def_values[p_name] = default

    @property
    def starting_point(self):
        """Starting point of the search space.
        """
        return (self.__def_values.values())

    @property
    def starting_point_asdict(self):
        return self.__def_values

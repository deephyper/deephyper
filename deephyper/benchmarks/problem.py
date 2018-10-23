from collections import OrderedDict

class Problem:
    """Representation of a problem.

    Attribute:
        space (OrderedDict): represents the search space of the problem.
    """

    def __init__(self):
        self.space = OrderedDict()

    def add_dim(self, p_name, p_value):
        """Add a dimension to the search space.

        Args:
            p_name (str): name of the parameter/dimension.
            p_space (Object): space corresponding to the new dimension.
        """
        self.space[p_name] = p_value

class HpProblem(Problem):
    def __init__(self):
        super().__init__()
        self.__def_values = OrderedDict()

    def add_dim(self, p_name, p_space, p_default=None):
        """Add a dimension to the search space.

        Args:
            p_name (str): name of the parameter/dimension.
            p_space (Object): space corresponding to the new dimension.
            p_default (): default value of the new dimension, it must be compatible with the p_space given.
        """
        assert type(p_name) is str, f'p_name must be str type, got {type(p_name)} !'
        assert type(p_space) is tuple or type(p_space) is list, f'p_space must be tuple or list type, got {type(p_space)} !'
        super().add_dim(p_name, p_space)
        self.__def_values[p_name] = p_default

    @property
    def starting_point(self):
        return (self.__def_values.values())

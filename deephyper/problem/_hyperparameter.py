import copy
from pprint import pformat

import ConfigSpace as cs
import ConfigSpace.hyperparameters as csh
import deephyper.core.exceptions as dh_exceptions
import numpy as np


def check_hyperparameter(parameter, name=None, default_value=None):
    """Check if the passed parameter is a valid description of an hyperparameter.

    :meta private:

    Args:
        parameter (str|Hyperparameter): a instance of ``ConfigSpace.hyperparameters.hyperparameter`` or a synthetic description (e.g., ``list``, ``tuple``).
        parameter (str): the name of the hyperparameter. Only required when the parameter is not a ``ConfigSpace.hyperparameters.hyperparameter``.
        default_value: a default value for the hyperparameter.

    Returns:
        Hyperparameter: the ConfigSpace hyperparameter instance corresponding to the ``parameter`` description.
    """
    if isinstance(parameter, csh.Hyperparameter):
        return parameter

    if not isinstance(parameter, (list, tuple, np.ndarray, dict)):
        raise ValueError(
            "Shortcut definition of an hyper-parameter has to be a list, tuple, array or dict."
        )

    if not (type(name) is str):
        raise ValueError("The 'name' of an hyper-parameter should be a string!")

    kwargs = {}
    if default_value is not None:
        kwargs["default_value"] = default_value

    if type(parameter) is tuple:  # Range of reals or integers
        if len(parameter) == 2:
            prior = "uniform"
        elif len(parameter) == 3:
            prior = parameter[2]
            assert prior in [
                "uniform",
                "log-uniform",
            ], f"Prior has to be 'uniform' or 'log-uniform' when {prior} was given for parameter '{name}'"
            parameter = parameter[:2]

        log = prior == "log-uniform"

        if all([isinstance(p, int) for p in parameter]):
            return csh.UniformIntegerHyperparameter(
                name=name, lower=parameter[0], upper=parameter[1], log=log, **kwargs
            )
        elif any([isinstance(p, float) for p in parameter]):
            return csh.UniformFloatHyperparameter(
                name=name, lower=parameter[0], upper=parameter[1], log=log, **kwargs
            )
    elif type(parameter) is list:  # Categorical
        if any(
            [isinstance(p, (str, bool)) or isinstance(p, np.bool_) for p in parameter]
        ):
            return csh.CategoricalHyperparameter(name, choices=parameter, **kwargs)
        elif all([isinstance(p, (int, float)) for p in parameter]):
            return csh.OrdinalHyperparameter(name, sequence=parameter)
    elif type(parameter) is dict: # Integer or Real distribution

        # Normal
        if "mu" in parameter and "sigma" in parameter:
            if type(parameter["mu"]) is float:
                return csh.NormalFloatHyperparameter(name=name, **parameter, **kwargs)
            elif type(parameter["mu"]) is int:
                return csh.NormalIntegerHyperparameter(name=name, **parameter, **kwargs)
            else:
                raise ValueError("Wrong hyperparameter definition! 'mu' should be either a float or an integer.")

    raise ValueError(
        f"Invalid dimension {name}: {parameter}. Read the documentation for"
        f" supported types."
    )


class HpProblem:
    """Class to define an hyperparameter problem.

    >>> from deephyper.problem import HpProblem
    >>> problem = HpProblem()

    Args:
        config_space (ConfigurationSpace, optional): In case the ``HpProblem`` is defined from a `ConfigurationSpace`.
    """

    def __init__(self, config_space=None):

        if config_space is not None and not (
            isinstance(config_space, cs.ConfigurationSpace)
        ):
            raise ValueError(
                "Parameter 'config_space' should be an instance of ConfigurationSpace!"
            )

        if config_space:
            self._space = copy.deepcopy(config_space)
        else:
            self._space = cs.ConfigurationSpace()
        self.references = []  # starting points

    def __str__(self):
        return repr(self)

    def __repr__(self):
        prob = repr(self._space)
        if len(self.references) == 0:
            return prob
        else:
            start_points = (
                f"{pformat({k:v for k,v in enumerate(self.starting_point_asdict)})}"
            )
            prob += "\n\n  Starting Point"
            prob += "s" if len(self.references) > 1 else ""
            prob += ":\n" + start_points
            return prob

    def add_hyperparameter(
        self, value, name: str = None, default_value=None
    ) -> csh.Hyperparameter:
        """Add an hyperparameter to the ``HpProblem``.

        Hyperparameters can be added to a ``HpProblem`` with a short syntax:

        >>> problem.add_hyperparameter((0, 10), "discrete", default_value=5)
        >>> problem.add_hyperparameter((0.0, 10.0), "real", default_value=5.0)
        >>> problem.add_hyperparameter([0, 10], "categorical", default_value=0)

        Sampling distributions can be provided:

        >>> problem.add_hyperparameter((0.0, 10.0, "log-uniform"), "real", default_value=5.0)

        It is also possible to use `ConfigSpace <https://automl.github.io/ConfigSpace/master/API-Doc.html#hyperparameters>`_ ``Hyperparameters``:

        >>> import ConfigSpace.hyperparameters as csh
        >>> csh_hp = csh.UniformIntegerHyperparameter(
        ...     name='uni_int', lower=10, upper=100, log=False)
        >>> problem.add_hyperparameter(csh_hp)

        Args:
            value (tuple or list or ConfigSpace.Hyperparameter): a valid hyperparametr description.
            name (str): The name of the hyperparameter to add.
            default_value (float or int or str): A default value for the corresponding hyperparameter.

        Returns:
            ConfigSpace.Hyperparameter: a ConfigSpace ``Hyperparameter`` object corresponding to the ``(value, name, default_value)``.
        """
        if not (type(name) is str or name is None):
            raise dh_exceptions.problem.SpaceDimNameOfWrongType(name)
        csh_parameter = check_hyperparameter(value, name, default_value=default_value)
        self._space.add_hyperparameter(csh_parameter)
        return csh_parameter

    def add_hyperparameters(self, hp_list):
        """Add a list of hyperparameters. It can be useful when a list of ``ConfigSpace.Hyperparameter`` are defined and we need to add them to the ``HpProblem``.

        Args:
            hp_list (ConfigSpace.Hyperparameter): a list of ConfigSpace hyperparameters.

        Returns:
            list: The list of added hyperparameters.
        """
        return [self.add_hyperparameter(hp) for hp in hp_list]

    def add_forbidden_clause(self, clause):
        """Add a `forbidden clause <https://automl.github.io/ConfigSpace/master/API-Doc.html#forbidden-clauses>`_ to the ``HpProblem``.

        For example if we want to optimize :math:`\\frac{1}{x}` where :math:`x` cannot be equal to 0:

        >>> from deephyper.problem import HpProblem
        >>> import ConfigSpace as cs
        >>> problem = HpProblem()
        >>> x = problem.add_hyperparameter((0.0, 10.0), "x")
        >>> problem.add_forbidden_clause(cs.ForbiddenEqualsClause(x, 0.0))

        Args:
            clause: a ConfigSpace forbidden clause.
        """
        self._space.add_forbidden_clause(clause)

    def add_condition(self, condition):
        """Add a `condition <https://automl.github.io/ConfigSpace/master/API-Doc.html#conditions>`_ to the ``HpProblem``.

        >>> from deephyper.problem import HpProblem
        >>> import ConfigSpace as cs
        >>> problem = HpProblem()
        >>> x = problem.add_hyperparameter((0.0, 10.0), "x")
        >>> y = problem.add_hyperparameter((1e-4, 1.0), "y")
        >>> problem.add_condition(cs.LessThanCondition(y, x, 1.0))

        Args:
            condition: A ConfigSpace condition.
        """
        self._space.add_condition(condition)

    def add_conditions(self, conditions: list) -> None:
        """Add a list of `condition <https://automl.github.io/ConfigSpace/master/API-Doc.html#conditions>`_ to the ``HpProblem``.

        Args:
            conditions (list): A list of ConfigSpace conditions.
        """
        self._space.add_conditions(conditions)

    @property
    def space(self):
        """The wrapped ConfigSpace object."""
        return self._space

    def add_starting_point(self, **parameters):
        """Add starting points to the ``HpProblem``. It is useful when a good-baseline is known to help initialize the search at a given location of the search space.

        >>> from deephyper.problem import HpProblem
        >>> problem = HpProblem()
        >>> x = problem.add_hyperparameter((0.0, 10.0), "x")
        >>> problem.add_starting_point(x=1.0)
        """
        self.check_configuration(parameters)
        self.references.append([parameters[p_name] for p_name in self._space])

    def check_configuration(self, parameters):
        """
        :meta private:
        """
        config = cs.Configuration(self._space, parameters)
        self._space.check_configuration(config)

    @property
    def starting_point(self):
        """Starting point(s) of the search space.

        Returns:
            list: list of starting points where each point is a list of values. Values are indexed in the same order as the order of creation of space's dimensions.
        """
        return self.references

    @property
    def starting_point_asdict(self):
        """Starting point(s) of the search space.

        Returns:
            list(dict): list of starting points where each point is a dict of values. Each key are correspnding to dimensions of the space.
        """
        return [{k: v for k, v in zip(list(self._space), p)} for p in self.references]

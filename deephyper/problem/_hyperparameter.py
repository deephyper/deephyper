import copy
import json

import ConfigSpace as cs
import ConfigSpace.hyperparameters as csh
import numpy as np
from ConfigSpace.read_and_write import json as cs_json

import deephyper.core.exceptions as dh_exceptions
import deephyper.skopt


def convert_to_skopt_dim(cs_hp, surrogate_model=None):

    if surrogate_model in ["RF", "ET", "GBRT"]:
        # models not sensitive to the metric space such as trees
        surrogate_model_type = "rule_based"
    else:
        # models sensitive to the metric space such as GP, neural networks
        surrogate_model_type = "distance_based"

    if isinstance(cs_hp, csh.UniformIntegerHyperparameter):
        skopt_dim = deephyper.skopt.space.Integer(
            low=cs_hp.lower,
            high=cs_hp.upper,
            prior="log-uniform" if cs_hp.log else "uniform",
            name=cs_hp.name,
        )
    elif isinstance(cs_hp, csh.UniformFloatHyperparameter):
        skopt_dim = deephyper.skopt.space.Real(
            low=cs_hp.lower,
            high=cs_hp.upper,
            prior="log-uniform" if cs_hp.log else "uniform",
            name=cs_hp.name,
        )
    elif isinstance(cs_hp, csh.CategoricalHyperparameter):
        # the transform is important if we don't want the complexity of trees
        # to explode with categorical variables
        skopt_dim = deephyper.skopt.space.Categorical(
            categories=cs_hp.choices,
            name=cs_hp.name,
            transform="onehot" if surrogate_model_type == "distance_based" else "label",
        )
    elif isinstance(cs_hp, csh.OrdinalHyperparameter):
        categories = list(cs_hp.sequence)
        if all(
            isinstance(x, (int, np.integer)) or isinstance(x, (float, np.floating))
            for x in categories
        ):
            transform = "identity"
        else:
            transform = "label"
        skopt_dim = deephyper.skopt.space.Categorical(
            categories=categories, name=cs_hp.name, transform=transform
        )
    else:
        raise TypeError(f"Cannot convert hyperparameter of type {type(cs_hp)}")

    return skopt_dim


def convert_to_skopt_space(cs_space, surrogate_model=None):
    """Convert a ConfigurationSpace to a scikit-optimize Space.

    Args:
        cs_space (ConfigurationSpace): the ``ConfigurationSpace`` to convert.
        surrogate_model (str, optional): the type of surrogate model/base estimator used to perform Bayesian optimization. Defaults to None.

    Raises:
        TypeError: if the input space is not a ConfigurationSpace.
        RuntimeError: if the input space contains forbiddens.
        RuntimeError: if the input space contains conditions

    Returns:
        deephyper.skopt.space.Space: a scikit-optimize Space.
    """

    # verify pre-conditions
    if not (isinstance(cs_space, cs.ConfigurationSpace)):
        raise TypeError("Input space should be of type ConfigurationSpace")

    if len(cs_space.get_conditions()) > 0:
        raise RuntimeError("Cannot convert a ConfigSpace with Conditions!")

    if len(cs_space.get_forbiddens()) > 0:
        raise RuntimeError("Cannot convert a ConfigSpace with Forbiddens!")

    # convert the ConfigSpace to deephyper.skopt.space.Space
    dimensions = []
    for hp in cs_space.get_hyperparameters():
        dimensions.append(convert_to_skopt_dim(hp, surrogate_model))

    skopt_space = deephyper.skopt.space.Space(dimensions)
    return skopt_space


def check_hyperparameter(parameter, name=None, default_value=None):
    """Check if the passed parameter is a valid description of an hyperparameter.

    :meta private:

    Args:
        parameter (str|Hyperparameter): an instance of ``ConfigSpace.hyperparameters.hyperparameter`` or a synthetic description (e.g., ``list``, ``tuple``).
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
            return csh.OrdinalHyperparameter(name, sequence=parameter, **kwargs)
    elif type(parameter) is dict:  # Integer or Real distribution

        # Normal
        if "mu" in parameter and "sigma" in parameter:
            if type(parameter["mu"]) is float:
                return csh.NormalFloatHyperparameter(name=name, **parameter, **kwargs)
            elif type(parameter["mu"]) is int:
                return csh.NormalIntegerHyperparameter(name=name, **parameter, **kwargs)
            else:
                raise ValueError(
                    "Wrong hyperparameter definition! 'mu' should be either a float or an integer."
                )

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

    @property
    def hyperparameter_names(self):
        """The list of hyperparameters names."""
        return self._space.get_hyperparameter_names()

    def check_configuration(self, parameters: dict):
        """Check if a configuration is valid. Raise an error if not."""
        config = cs.Configuration(self._space, parameters)
        self._space.check_configuration(config)

    @property
    def default_configuration(self):
        """The default configuration as a dictionnary."""
        config = self._space.get_default_configuration().get_dictionary()
        return config

    def to_json(self):
        """Returns a dict version of the space which can be saved as JSON."""
        json_format = json.loads(cs_json.write(self._space))
        return json_format

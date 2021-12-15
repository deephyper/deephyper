import copy
from pprint import pformat

import cconfigspace as ccs
import deephyper.core.exceptions as dh_exceptions
import skopt
import numpy as np


# def convert_to_skopt_dim(cs_hp, surrogate_model=None):

#     if surrogate_model in ["RF", "ET", "GBRT"]:
#         # models not sensitive ot the metric space such as trees
#         surrogate_model_type = "rule_based"
#     else:
#         # models sensitive to the metric space such as GP, neural networks
#         surrogate_model_type = "distance_based"

#     if isinstance(cs_hp, csh.UniformIntegerHyperparameter):
#         skopt_dim = skopt.space.Integer(
#             low=cs_hp.lower,
#             high=cs_hp.upper,
#             prior="log-uniform" if cs_hp.log else "uniform",
#             name=cs_hp.name,
#         )
#     elif isinstance(cs_hp, csh.UniformFloatHyperparameter):
#         skopt_dim = skopt.space.Real(
#             low=cs_hp.lower,
#             high=cs_hp.upper,
#             prior="log-uniform" if cs_hp.log else "uniform",
#             name=cs_hp.name,
#         )
#     elif isinstance(cs_hp, csh.CategoricalHyperparameter):
#         # the transform is important if we don't want the complexity of trees
#         # to explode with categorical variables
#         skopt_dim = skopt.space.Categorical(
#             categories=cs_hp.choices,
#             name=cs_hp.name,
#             transform="onehot" if surrogate_model_type == "distance_based" else "label",
#         )
#     elif isinstance(cs_hp, csh.OrdinalHyperparameter):
#         skopt_dim = skopt.space.Categorical(
#             categories=list(cs_hp.sequence), name=cs_hp.name, transform="label"
#         )
#     else:
#         raise TypeError(f"Cannot convert hyperparameter of type {type(cs_hp)}")

#     return skopt_dim


# def convert_to_skopt_space(cs_space, surrogate_model=None):
#     """Convert a ConfigurationSpace to a scikit-optimize Space.

#     Args:
#         cs_space (ConfigurationSpace): the ``ConfigurationSpace`` to convert.
#         surrogate_model (str, optional): the type of surrogate model/base estimator used to perform Bayesian optimization. Defaults to None.

#     Raises:
#         TypeError: if the input space is not a ConfigurationSpace.
#         RuntimeError: if the input space contains forbiddens.
#         RuntimeError: if the input space contains conditions

#     Returns:
#         skopt.space.Space: a scikit-optimize Space.
#     """

#     # verify pre-conditions
#     if not (isinstance(cs_space, cs.ConfigurationSpace)):
#         raise TypeError("Input space should be of type ConfigurationSpace")

#     if len(cs_space.get_conditions()) > 0:
#         raise RuntimeError("Cannot convert a ConfigSpace with Conditions!")

#     if len(cs_space.get_forbiddens()) > 0:
#         raise RuntimeError("Cannot convert a ConfigSpace with Forbiddens!")

#     # convert the ConfigSpace to skopt.space.Space
#     dimensions = []
#     for hp in cs_space.get_hyperparameters():
#         dimensions.append(convert_to_skopt_dim(hp, surrogate_model))

#     skopt_space = skopt.space.Space(dimensions)
#     return skopt_space


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
    if isinstance(parameter, ccs.Hyperparameter):
        return parameter

    if not isinstance(parameter, (list, tuple, np.ndarray, dict)):
        raise ValueError(
            "Shortcut definition of an hyper-parameter has to be a list, tuple, array or dict."
        )

    if not (type(name) is str):
        raise ValueError("The 'name' of an hyper-parameter should be a string!")

    ccs_dist = None

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

        if all([isinstance(p, int) for p in parameter]):  # integer
            ccs_param = ccs.NumericalHyperparameter.int(
                name=name, lower=parameter[0], upper=parameter[1], default=default_value
            )

            if prior == "log-uniform":
                ccs_dist = ccs.UniformDistribution.int(
                    lower=parameter[0],
                    upper=parameter[1],
                    scale=ccs.ccs_scale_type.LOGARITHMIC,
                )

        elif any([isinstance(p, float) for p in parameter]):
            ccs_param = ccs.NumericalHyperparameter.float(
                name=name, lower=parameter[0], upper=parameter[1], default=default_value
            )

            if prior == "log-uniform":
                ccs_dist = ccs.UniformDistribution.float(
                    lower=parameter[0],
                    upper=parameter[1],
                    scale=ccs.ccs_scale_type.LOGARITHMIC,
                )

        return ccs_param, ccs_dist

    elif type(parameter) is list:  # Categorical
        default_index = 0
        if default_value:
            default_index = parameter.index(default_value)
        if any(
            [isinstance(p, (str, bool)) or isinstance(p, np.bool_) for p in parameter]
        ):
            ccs_param = ccs.CategoricalHyperparameter(
                name=name, values=parameter, default_index=default_index
            )
        elif all([isinstance(p, (int, float)) for p in parameter]):

            ccs_param = ccs.OrdinalHyperparameter(
                name=name, values=parameter, default_index=default_index
            )

        return ccs_param, ccs_dist
        
    # elif type(parameter) is dict:  # Integer or Real distribution

    #     # Normal
    #     if "mu" in parameter and "sigma" in parameter:
    #         if type(parameter["mu"]) is float:
    #             return csh.NormalFloatHyperparameter(name=name, **parameter, **kwargs)
    #         elif type(parameter["mu"]) is int:
    #             return csh.NormalIntegerHyperparameter(name=name, **parameter, **kwargs)
    #         else:
    #             raise ValueError(
    #                 "Wrong hyperparameter definition! 'mu' should be either a float or an integer."
    #             )

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
            isinstance(config_space, ccs.ConfigurationSpace)
        ):
            raise ValueError(
                "Parameter 'config_space' should be an instance of ConfigurationSpace!"
            )

        if config_space:
            self._space = copy.deepcopy(config_space)
        else:
            self._space = ccs.ConfigurationSpace()
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
    ) -> ccs.Hyperparameter:
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
        ccs_param, ccs_dist = check_hyperparameter(
            value, name, default_value=default_value
        )
        self._space.add_hyperparameter(ccs_param)
        if ccs_dist:
            self._space.set_distribution(ccs_dist, [ccs_param])
        return ccs_param

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
        return [hp.name for hp in self._space.hyperparameters]

    def add_starting_point(self, **parameters):
        """Add starting points to the ``HpProblem``. It is useful when a good-baseline is known to help initialize the search at a given location of the search space.

        >>> from deephyper.problem import HpProblem
        >>> problem = HpProblem()
        >>> x = problem.add_hyperparameter((0.0, 10.0), "x")
        >>> problem.add_starting_point(x=1.0)
        """
        self.check_configuration(parameters)
        self.references.append([parameters[p_name] for p_name in self.hyperparameter_names])

    def check_configuration(self, parameters: dict):
        """
        :meta private:
        """
        values = [parameters[hp_name] for hp_name in self.hyperparameter_names]
        config = ccs.Configuration(configuration_space=self._space, values=values)
        config.check()
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

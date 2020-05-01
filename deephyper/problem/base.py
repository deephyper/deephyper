import inspect
from collections import OrderedDict
from inspect import signature
from pprint import pformat

import ConfigSpace as cs
import ConfigSpace.hyperparameters as csh
import numpy as np

import deephyper.core.exceptions as dh_exceptions


def check_hyperparameter(parameter, name=None):
    if isinstance(parameter, csh.Hyperparameter):
        return parameter

    if not isinstance(parameter, (list, tuple, np.ndarray)):
        raise ValueError(
            "Shortcut definition of an hyper-parameter has to be a list, tuple, array."
        )

    if not (type(name) is str):
        raise dh_exceptions.problem.SpaceDimNameOfWrongType(name)

    # if isinstance(parameter, (int, float)):  # Constant parameter
    #     return csh.Constant(name, parameter)

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
                name=name, lower=parameter[0], upper=parameter[1], log=log
            )
        elif any([isinstance(p, float) for p in parameter]):
            return csh.UniformFloatHyperparameter(
                name=name, lower=parameter[0], upper=parameter[1], log=log
            )
    elif type(parameter) is list:  # Categorical
        if any(
            [isinstance(p, (str, bool)) or isinstance(p, np.bool_) for p in parameter]
        ):
            return csh.CategoricalHyperparameter(name, choices=parameter)
        elif all([isinstance(p, (int, float)) for p in parameter]):
            return csh.OrdinalHyperparameter(name, sequence=parameter)

    raise ValueError(
        f"Invalid dimension {name}: {parameter}. Read the documentation for"
        f" supported types."
    )


class BaseProblem:
    """Representation of a problem.
    """

    def __init__(self, config_space=None, seed=42):
        if config_space is None:
            self.seed = seed
            self._space = cs.ConfigurationSpace(seed=seed)
        else:
            self._space = config_space
            if seed is None:
                self.seed = self._space.random.get_state()[1][0]
            else:
                self.seed = seed
                self._space.seed(seed)
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

    def add_dim(self, p_name=None, p_space=None):
        """Deprecated! Add a dimension to the search space.

        Args:
            p_name (str): name of the parameter/dimension.
            p_space (Object): space corresponding to the new dimension.
        """

        return self.add_hyperparameter(p_space, p_name)

    def add_hyperparameter(self, value, name: str = None) -> csh.Hyperparameter:
        """Add an hyperparameter to the search space of the Problem.

        Args:
            name (str): The name of the hyper-parameter
            value (tuple or list or ConfigSpace.Hyperparameter): [description]

        Returns:
            csh.Hyperparameter: a ConfigSpace.Hyperparameter object corresponding to the (name, value).
        """
        if not (type(name) is str or name is None):
            raise dh_exceptions.problem.SpaceDimNameOfWrongType(name)
        csh_parameter = check_hyperparameter(value, name)
        self._space.add_hyperparameter(csh_parameter)
        return csh_parameter

    def add_forbidden_clause(self, clause):
        self._space.add_forbidden_clause(clause)

    def add_condition(self, condition):
        self._space.add_condition(condition)

    @property
    def space(self):
        return self._space

    def add_starting_point(self, **parameters):
        self.check_configuration(parameters)
        self.references.append([parameters[p_name] for p_name in self._space])

    def check_configuration(self, parameters):
        config = cs.Configuration(self._space, parameters)
        self._space.check_configuration(config)

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
        return [{k: v for k, v in zip(list(self._space), p)} for p in self.references]


def test_base_problem():
    import ConfigSpace.hyperparameters as CSH

    alpha = CSH.UniformFloatHyperparameter(name="alpha", lower=0, upper=1)
    beta = CSH.UniformFloatHyperparameter(name="beta", lower=0, upper=1)

    pb = BaseProblem(42)
    pb.add_hyperparameter(alpha)
    pb.add_hyperparameter(beta)

    print(pb)


if __name__ == "__main__":
    test_base_problem()

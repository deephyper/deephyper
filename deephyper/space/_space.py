"""
Documentation on discrete latent variable enumeration in Pyro: http://pyro.ai/examples/enumeration.html
"""

import abc
from collections import OrderedDict

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import HMC, MCMC, MixedHMC
from typing import Union, List, Dict, Any

from deephyper.space._constraint import BooleanConstraint


class Space:
    """Used to define a search space composed of dimensions, distributions
    and contraints.

    Args:
        name (str): the name of the search space.
        seed (int, optional): the random seed of the search space. Defaults to ``None``.
    """

    def __init__(self, name: str, seed: int = None):

        self.name = name
        self.dimensions = OrderedDict()
        self.constraints = OrderedDict()
        self._mcmc = None
        if seed:
            self._key = jax.random.key(seed)
        else:
            self._key = jax.random.key(
                np.random.default_rng().integers(np.iinfo(np.int32).max)
            )

    def __repr__(self) -> str:
        out = f"Space: {self.name}\n"

        # Dimensions
        out += "  Dimensions:\n"
        for dim in self.dimensions.values():
            out += f"  - {dim}\n"

        # Constraints
        if len(self.constraints) > 0:
            out += "  Constraints:\n"
            for cons in self.constraints.values():
                out += f"  - {cons}\n"
        return out

    def add_dimension(self, *args):
        """Add a dimension to the space.

        Args:
            *args : iterable of Dimension.

        Returns:
            self: the instance of the space.
        """
        for dim in args:
            self.dimensions[dim.name] = dim
        return self

    def add_constraint(self, *args):
        for cons in args:
            self.constraints[cons.name] = cons
        return self

    @property
    def default_value(self):
        return [v.default_value for v in self.dimensions.values()]

    @default_value.setter
    def default_value(self, values):
        # Set default values by name of the dimension
        if isinstance(values, dict):
            for name, v in values.items():
                self.dimensions[name].value = v

        # Set default values by order of the dimensions
        else:
            for dim, v in zip(self.dimensions.values(), values):
                dim.value = v

    @property
    def is_mixed(self) -> bool:
        """Return True if the space contains mixed types of dimensions."""
        return any(
            [
                isinstance(dim, (CatDimension, IntDimension))
                for dim in self.dimensions.values()
            ]
        )

    @property
    def is_constant(self) -> bool:
        """Return True if the space contains only constant dimensions."""
        return all(
            [isinstance(dim, ConstDimension) for dim in self.dimensions.values()]
        )

    def is_feasible(self, parameters: Union[List, Dict]) -> bool:
        """Check if the parameters are feasible with respect to the constraints."""
        if isinstance(parameters, list):
            parameters = {
                name: value for name, value in zip(self.dimensions.keys(), parameters)
            }
        elif isinstance(parameters, dict):
            pass
        else:
            raise ValueError(
                f"parameters should be a list or a dict, got {type(parameters)}"
            )
        for cons in self.constraints.values():
            if not cons.is_feasible(parameters):
                return False
        return True

    def sample(self, num_samples: int = None):
        # TODO: parallelize with multiple chains on different devices
        num_chains = 1

        def model():

            # Sample each dimension
            parameters = {}
            for name, dim in self.dimensions.items():
                if isinstance(dim, ConstDimension):
                    # More details about Delta distribution used in ConstDimension
                    # url: https://forum.pyro.ai/t/cannot-find-valid-initial-parameters-with-delta-distribution/1636
                    numpyro.deterministic(name, 0.0)
                elif isinstance(dim, IntDimension):
                    parameters[name] = numpyro.sample(name, dim.distribution)

                else:
                    parameters[name] = numpyro.sample(name, dim.distribution)

            # Enforce constrains
            # Continuous constraints
            for cons in self.constraints.values():
                if not (isinstance(cons, BooleanConstraint)):
                    numpyro.factor(cons.name, cons(parameters))

            # Boolean constraints
            # !v1: all boolean constraints clauses are merged in one  SAT formula
            # !often returns 0 solutions
            # is_valid = jnp.ones((num_samples,), dtype=bool)
            # for cons in self.constraints.values():
            #     if isinstance(cons, BooleanConstraint):
            #         is_valid = is_valid & cons.is_feasible(parameters)
            # numpyro.factor(
            #     "boolean_constraint_is_valid", 10 * jnp.where(is_valid, 1, 0)
            # )

            # !v2: each boolean constraint clause is mapped to a log-likelihood factor
            for cons in self.constraints.values():
                if isinstance(cons, BooleanConstraint):
                    is_valid = cons.is_feasible(parameters)
                    numpyro.factor(
                        cons.name,
                        cons.strength * jnp.where(is_valid, 1, 0),
                    )

        if self.is_constant:
            samples = OrderedDict()
            for name, dim in self.dimensions.items():
                samples[name] = np.full((num_samples,), dim.value, dtype="O")
        else:
            if self._mcmc is None:
                kernel = HMC(model, trajectory_length=1.5)
                if self.is_mixed:
                    kernel = MixedHMC(
                        kernel,
                        num_discrete_updates=None,
                        random_walk=False,
                        modified=False,
                    )
                self._mcmc = MCMC(
                    kernel,
                    num_warmup=1_000,
                    num_chains=num_chains,
                    num_samples=num_samples // num_chains,
                    progress_bar=True,
                )
            self._key, subkey = jax.random.split(self._key)
            self._mcmc.run(subkey)
            samples = self._mcmc.get_samples()

        # Checking which constraint are to be strictly enforced
        feasible_indexes = np.ones(
            (num_samples // num_chains * num_chains,), dtype=bool
        )
        for name, cons in self.constraints.items():
            if hasattr(cons, "is_strict") and cons.is_strict:
                feasible_indexes = feasible_indexes & cons.is_feasible(samples)

        samples = np.asarray(
            [samples[name] for name in self.dimensions.keys()], dtype="O"
        )

        for i, dim in enumerate(self.dimensions.values()):
            if isinstance(dim, (CatDimension, ConstDimension)):
                samples[i, :] = dim.inverse(samples[i, :])
            elif isinstance(dim, IntDimension):
                samples[i, :] = np.floor(samples[i, :]).astype(np.int32)

        return samples.T[feasible_indexes].tolist()


class Dimension(abc.ABC):
    """Used to define a dimension of the search space.

    Args:
        name (str): the name of the dimension.
        default_value (Any, optional): The default value of the dimension. Defaults to ``None``.
    """

    def __init__(self, name: str, default_value=None):

        self.name = name
        self.default_value = (
            default_value  # Default value of a variable from this dimension.
        )
        self.distribution = (
            None  # Default distribution of a variable from this dimension.
        )
        self.inactive_value = (
            default_value  # Slack value when the variable is considered inactive.
        )

    def __repr__(self) -> str:
        return f"{self.name} ({self.default_value})"


class IntDimension(Dimension):
    """Used to define a discrete dimension of the search space.

    Args:
        name (str): the name of the dimension.
        low (float): the lower bound of the dimension.
        high (float): the upper bound of the dimension.
        default_value (float, optional): The default value of the dimension. Defaults to ``None`` will attribute the lower bound as ``default_value``.
    """

    def __init__(self, name: str, low: int, high: int, default_value: int = None):
        super().__init__(name, default_value)
        self.low = int(low)
        self.high = int(high)
        if self.default_value is None:
            self.default_value = self.low
        self.distribution = dist.DiscreteUniform(self.low, self.high)
        self.inactive_value = self.low

    def __repr__(self) -> str:
        out = super().__repr__()
        out += f" in ({self.low}, {self.high})"
        return out


class RealDimension(Dimension):
    """Used to define a real dimension of the search space.

    Args:
        name (str): the name of the dimension.
        low (float): the lower bound of the dimension.
        high (float): the upper bound of the dimension.
        default_value (float, optional): the default value of the dimension. Defaults to ``None`` will attribute the first categorical value of the list as ``default_value``.
    """

    def __init__(self, name: str, low: float, high: float, default_value: float = None):
        super().__init__(name, default_value)
        self.low = float(low)
        self.high = float(high)
        if self.default_value is None:
            self.default_value = self.low
        self.distribution = dist.Uniform(low=self.low, high=self.high)
        self.inactive_value = self.low

    def __repr__(self) -> str:
        out = super().__repr__()
        out += f" in ({self.low}, {self.high})"
        return out


class CatDimension(Dimension):
    """Used to defined a categorical dimension of the search space.

    Args:
        name (str): the name of the dimension.
        categories (list): the list of categorical values.
        default_value (_type_, optional): the default value of the dimension. Defaults to None.
        ordered (bool, optional): _description_. Defaults to False.
    """

    def __init__(
        self, name: str, categories: list, default_value=None, ordered: bool = False
    ):
        super().__init__(name, default_value)
        self.categories = list(categories)
        self.encoding = {k: i for i, k in enumerate(self.categories)}
        self.ordered = ordered
        if self.default_value is None:
            self.default_value = self.categories[0]
        self.distribution = dist.Categorical(
            probs=np.ones(len(categories)) / len(categories)
        )
        self.inactive_value = self.categories[0]

    def __repr__(self) -> str:
        out = super().__repr__()
        out += f" in {self.categories}"
        return out

    def inverse(self, value):
        """Transform a value from the sampling space (indexes) to the original space."""
        # Check if the value is iterable
        value_is_iterable = True
        try:
            iter(value)
        except TypeError:
            value_is_iterable = False
        if value_is_iterable:
            return [self.categories[v] for v in value]
        else:
            return self.categories[value]

    def transform(self, value):
        """Transform a value from the original space to the sampling space (indexes)."""
        # Check if the value is iterable
        value_is_iterable = True
        try:
            iter(value)
        except TypeError:
            value_is_iterable = False
        if value_is_iterable:
            return [self.encoding[v] for v in value]
        else:
            return self.encoding[value]


class ConstDimension(Dimension):
    """Used to defined a constant dimension of the search space.

    Args:
        name (str): the name of the dimension.
        value (Any): the constant value.
    """

    def __init__(self, name: str, value: Any):
        super().__init__(name, default_value=value)
        self.distribution = dist.Delta()
        self.value = value

    def __repr__(self) -> str:
        out = super().__repr__()
        out += " in {" + f"{self.value}" + "}"
        return out

    def inverse(self, value):
        """Transform a value from the sampling space (indexes) to the original space."""
        # Check if the value is iterable
        value_is_iterable = True
        try:
            iter(value)
        except TypeError:
            value_is_iterable = False
        if value_is_iterable:
            return [self.value for _ in value]
        else:
            return self.value

    def transform(self, value):
        """Transform a value from the original space to the sampling space (indexes)."""
        # Check if the value is iterable
        value_is_iterable = True
        try:
            iter(value)
        except TypeError:
            value_is_iterable = False
        if value_is_iterable:
            return [0.0 for _ in value]
        else:
            return 0.0

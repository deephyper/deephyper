"""
Documentation on discrete latent variable enumeration in Pyro: http://pyro.ai/examples/enumeration.html
"""

import abc
from collections import OrderedDict

import numpy as np
import numpyro.distributions as dist

import jax
import numpyro
from numpyro.infer import HMC, MCMC, MixedHMC


class Space(abc.ABC):
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

    def sample(self, num_samples: int = None):
        def model():

            # Sample each dimension
            parameters = {}
            for name, dim in self.dimensions.items():
                if isinstance(dim, ConstDimension):
                    # More details about Delta distribution used in ConstDimension
                    # url: https://forum.pyro.ai/t/cannot-find-valid-initial-parameters-with-delta-distribution/1636
                    numpyro.deterministic(name, 0.0)
                else:
                    parameters[name] = numpyro.sample(name, dim.distribution)

            # Enforce constrains
            for name, cons in self.constraints.items():
                numpyro.factor(name, cons(parameters))

        if self.is_constant:
            samples = OrderedDict()
            for name, dim in self.dimensions.items():
                samples[name] = np.full((num_samples,), dim.value, dtype="O")
        else:
            if self._mcmc is None:
                kernel = HMC(model, trajectory_length=1.2)
                if self.is_mixed:
                    kernel = MixedHMC(kernel, num_discrete_updates=20)
                self._mcmc = MCMC(
                    kernel, num_warmup=1000, num_samples=num_samples, progress_bar=False
                )
            self._key, subkey = jax.random.split(self._key)
            self._mcmc.run(subkey)
            samples = self._mcmc.get_samples()

        # Checking which constraint are to be strictly enforced
        feasible_indexes = np.ones((num_samples,), dtype=bool)
        for name, cons in self.constraints.items():
            if hasattr(cons, "is_strict") and cons.is_strict:
                feasible_indexes = np.logical_and(
                    feasible_indexes, cons.is_feasible(samples)
                )

        samples = np.asarray(
            [samples[name] for name in self.dimensions.keys()], dtype="O"
        )

        for i, dim in enumerate(self.dimensions.values()):
            if isinstance(dim, (CatDimension, ConstDimension)):
                samples[i, :] = dim.inverse(samples[i, :])

        return samples.T[feasible_indexes].tolist()


class Dimension(abc.ABC):  #
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
    def __init__(self, name: str, value):
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

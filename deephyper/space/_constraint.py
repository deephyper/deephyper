import abc
import jax.numpy as jnp


class Constraint:
    def __init__(self, name: str):
        self.name = name

    def __repr__(self) -> str:
        return f"s.t. {self.name}"

    @abc.abstractmethod
    def __call__(self, parameters: dict):
        raise NotImplementedError

    @abc.abstractmethod
    def is_feasible(self, parameters: dict) -> bool:
        raise NotImplementedError


class BooleanConstraint(Constraint):
    def __init__(self, name: str, func, strength: float = 10, is_strict: bool = False):
        super().__init__(name)
        self.name = name
        self.func = func
        self.strength = strength
        self.is_strict = is_strict

    def __call__(self, parameters: dict) -> bool:
        return -self.strength * jnp.where(self.func(parameters), 0, 1)

    def is_feasible(self, parameters: dict) -> bool:
        return self.func(parameters)


class InequalityConstraint(Constraint):
    """
    A constraint of the form f(x) <= 0
    """

    def __init__(self, name: str, func, strength: float = 10, is_strict: bool = False):
        super().__init__(name)
        self.func = func
        self.strength = strength
        self.is_strict = is_strict

    def __repr__(self) -> str:
        out = super().__repr__()
        out += " <= 0"
        if self.is_strict:
            out += " (strict)"
        return out

    def __call__(self, parameters: dict) -> float:
        return -jnp.exp(self.strength * self.func(parameters)) - 1

    def is_feasible(self, parameters: dict) -> bool:
        return self.func(parameters) <= 0


class EqualityConstraint(Constraint):
    def __init__(self, name: str, func, strength: float = 10, is_strict: bool = False):
        super().__init__(name)
        self.func = func
        self.strength = strength
        self.is_strict = is_strict

    def __repr__(self) -> str:
        out = super().__repr__()
        out += " <= 0"
        if self.is_strict:
            out += " (strict)"
        return out

    def __call__(self, parameters: dict) -> float:
        x = self.strength * self.func(parameters)
        return -jnp.exp(x) - jnp.exp(-x) - 2

    def is_feasible(self, parameters: dict) -> bool:
        value = self.func(parameters)
        return -1e-6 <= value and value <= 0 + 1e-6

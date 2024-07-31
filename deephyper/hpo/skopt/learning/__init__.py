"""Machine learning extensions for model-based optimization."""

from .forest import RandomForestRegressor
from .forest import ExtraTreesRegressor
from .gaussian_process import GaussianProcessRegressor
from .gbrt import GradientBoostingQuantileRegressor


__all__ = [
    "RandomForestRegressor",
    "ExtraTreesRegressor",
    "GradientBoostingQuantileRegressor",
    "GaussianProcessRegressor",
]

try:
    from skgarden.mondrian import MondrianForestRegressor  # noqa: F401

    __all__.append("MondrianForestRegressor")
except ImportError:
    pass

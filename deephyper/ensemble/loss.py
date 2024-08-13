import abc

import numpy as np
import scipy.stats as ss


class Loss(abc.ABC):
    """Represents a loss function for ensembles."""

    @abc.abstractmethod
    def __call__(self, y_true, y_pred):
        pass


class SquaredError(Loss):
    """The usual square loss ``(y_true - y_pred)**2`` used to estimate ``E[Y|X=x]``."""

    def __call__(self, y_true, y_pred):
        if isinstance(y_pred, dict):
            y_pred = y_pred["loc"]
        return np.square(y_true - y_pred)


class AbsoluteError(Loss):
    """The usual absolute loss ``(y_true - y_pred)**2`` used to estimate the median of ``P(Y|X=x)``."""

    def __call__(self, y_true, y_pred):
        if isinstance(y_pred, dict):
            y_pred = y_pred["loc"]
        return np.abs(y_true - y_pred)


class NormalNegLogLikelihood(Loss):

    def __init__(self):
        self.dist = ss.norm

    def __call__(self, y_true, y_pred):
        return -self.dist.logpdf(y_true, loc=y_pred["loc"], scale=y_pred["scale"])

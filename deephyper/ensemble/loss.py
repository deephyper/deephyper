import abc

import scipy.stats as ss


class Loss(abc.ABC):

    @abc.abstractmethod
    def __call__(self, y_true, y_pred):
        pass


class SquaredError(Loss):

    def __call__(self, y_true, y_pred):
        return ((y_true - y_pred) ** 2).mean()


class NormalNegLogLikelihood(Loss):

    def __init__(self):
        self.dist = ss.norm

    def __call__(self, y_true, y_pred):
        nll = -self.dist.logpdf(y_true, loc=y_pred["loc"], scale=y_pred["scale"]).mean()
        return nll

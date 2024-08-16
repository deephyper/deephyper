import abc
from typing import Callable, Sequence

from deephyper.ensemble.loss import Loss


class Selector(abc.ABC):
    """Base class that represents an algorithm that select a subset of predictors from a set of available predictors in order to build an ensemble.

    Args:
        loss_func (Callable or Loss): a loss function that takes two arguments: the true target values and the predicted target values.
    """

    def __init__(self, loss_func: Callable | Loss):
        self.loss_func = loss_func

    @abc.abstractmethod
    def select(self, y, y_predictors) -> Sequence[int]:
        """The selection algorithms.

        Args:
            y (np.ndarray): the true target values.
            y_predictors (_type_): a sequence of predictions from available predictors. It should be a list of length ``n_predictors`` with each element being the prediction of a predictor.

        Returns:
            Sequence[int]: the sequence of selected predictors.
        """
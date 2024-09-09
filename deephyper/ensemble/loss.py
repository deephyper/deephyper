"""Subpackage for loss functions of ensemble models.
"""

import abc

import numpy as np
import scipy.stats as ss


class Loss(abc.ABC):
    """Base class that represents the loss function of an ensemble."""

    @abc.abstractmethod
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray | dict) -> np.ndarray:
        """Compute the loss function.

        Args:
            y_true (np.ndarray): the true target values.
            y_pred (np.ndarray or dict): the predicted target values.

        Returns:
            np.ndarray: the loss value with first dimension ``n_samples``.
        """


class SquaredError(Loss):
    """The usual square loss ``(y_true - y_pred)**2`` used to estimate ``E[Y|X=x]``."""

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray | dict) -> np.ndarray:
        # If the prediction is a dictionary, we assume that the prediction is a distribution and we take the mean as prediction.
        if isinstance(y_pred, dict):
            y_pred = y_pred["loc"]
        return np.square(y_true - y_pred)


class AbsoluteError(Loss):
    """The usual absolute loss ``(y_true - y_pred)**2`` used to estimate the median of ``P(Y|X=x)``."""

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray | dict) -> np.ndarray:
        # If the prediction is a dictionary, we assume that the prediction is a distribution and we take the mean as prediction.
        if isinstance(y_pred, dict):
            y_pred = y_pred["loc"]
        return np.abs(y_true - y_pred)


class NormalNegLogLikelihood(Loss):
    """The negative log-likelihood of observed data ``y_true`` given predicted parameters of a normal distribution ``y_pred["loc"], y_pred["scale"]``."""

    def __init__(self):
        self.dist = ss.norm

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray | dict) -> np.ndarray:
        return -self.dist.logpdf(y_true, loc=y_pred["loc"], scale=y_pred["scale"])


class ZeroOneLoss(Loss):
    """Zero-One loss for classification (a.k.a, error rate) which is ``1`` if the prediction is wrong and ``0`` if it is correct.

    - ``y_true`` is an array of integers representing the true class labels.

    Args:
        predict_proba (bool, optional): A boolean indicating if ``y_pred`` contains predicted categorical probabilities. Defaults to ``False`` for label predictions.
    """

    def __init__(self, predict_proba: bool = False):
        self._predict_proba = predict_proba

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray | dict) -> np.ndarray:
        if isinstance(y_pred, dict):
            y_pred = y_pred["loc"]
        if self._predict_proba:
            return np.array(y_true != np.argmax(y_pred, axis=-1), dtype=float)
        else:
            return np.array(y_true != y_pred, dtype=float)


class CategoricalCrossEntropy(Loss):
    """Categorical-Cross Entropy (a.k.a., Log-Loss) function for classification.

    - ``y_true`` is an array of integers representing the true class labels.
    - ``y_pred`` is an array of predicted categorical probabilities.

    """

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray | dict) -> np.ndarray:
        if isinstance(y_pred, dict):
            y_pred = y_pred["loc"]
        prob = y_pred[np.arange(len(y_pred)), y_true]
        eps = np.finfo(prob.dtype).eps
        prob = np.clip(prob, eps, 1 - eps)
        return -np.log(prob)

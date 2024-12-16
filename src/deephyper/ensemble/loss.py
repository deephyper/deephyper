"""Subpackage for loss functions of ensemble models."""

import abc
from typing import Any, Dict

import numpy as np
import scipy.stats as ss


def _check_is_array(y: Any, y_name: str) -> None:
    """Verify that the passed value is of type np.ndarray and raise a ValueError otherwise.

    Args:
        y (Any): the value to verify.
        y_name (str): the name of the value to use in the raise error.

    Raises:
        ValueError: when ``y`` is not an np.ndarray.
    """
    if not isinstance(y, np.ndarray):
        raise ValueError(f"{y_name} should be of type np.ndarray but is of type {type(y)}")


def _check_is_array_or_dict(y: Any, y_name: str):
    """Verify that the passed value is of type np.ndarray or dict and raise a ValueError otherwise.

    Args:
        y (Any): the value to verify.
        y_name (str): the name of the value to use in the raise error.

    Raises:
        ValueError: when ``y`` is not an np.ndarray and not a dict.
    """
    if not isinstance(y, np.ndarray) and not isinstance(y, dict):
        raise ValueError(f"{y_name} should be of type np.ndarray or dict but is of type {type(y)}")


class Loss(abc.ABC):
    """Base class that represents the loss function of an ensemble.

    Losses represent functions that should be minimized.
    """

    @abc.abstractmethod
    def __call__(
        self, y_true: np.ndarray, y_pred: np.ndarray | Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Compute the loss function.

        Args:
            y_true (np.ndarray): the true target values.
            y_pred (np.ndarray or Dict[str, np.ndarray]): the predicted target values.

        Returns:
            np.ndarray: the loss value with first dimension ``n_samples``.
        """


class SquaredError(Loss):
    """The usual square loss ``(y_true - y_pred)**2`` used to estimate ``E[Y|X=x]``."""

    def __call__(
        self, y_true: np.ndarray, y_pred: np.ndarray | Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Compute the squared error.

        Args:
            y_true (np.ndarray): the true target values.
            y_pred (np.ndarray or Dict[str, np.ndarray]): the predicted target values. If it is a
            _Dict[str, np.ndarray]_ then it should contain a key ``"loc"``.

        Returns:
            np.ndarray: the loss value with first dimension ``n_samples``.
        """
        _check_is_array(y_true, "y_true")
        _check_is_array_or_dict(y_pred, "y_pred")

        # If the prediction is a dictionary, we assume that the prediction are distribution
        # parameters and we take the mean estimate that should correspond to the 'loc' key
        if isinstance(y_pred, dict):
            if "loc" not in y_pred:
                raise ValueError("y_pred should contain a 'loc' key when it is a dict")

            y_pred = y_pred["loc"]

        return np.square(y_true - y_pred)


class AbsoluteError(Loss):
    """The usual absolute loss ``(y_true - y_pred)**2``.

    It is used to estimate the median of ``P(Y|X=x)``.
    """

    def __call__(
        self, y_true: np.ndarray, y_pred: np.ndarray | Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Compute the absolute error.

        Args:
            y_true (np.ndarray): the true target values.
            y_pred (np.ndarray or Dict[str, np.ndarray]): the predicted target values. If it is a
            _Dict[str, np.ndarray]_ then it should contain a key ``"loc"``.

        Returns:
            np.ndarray: the loss value with first dimension ``n_samples``.
        """
        _check_is_array(y_true, "y_true")
        _check_is_array_or_dict(y_pred, "y_pred")

        # If the prediction is a dictionary, we assume that the prediction are distribution
        # parameters and we take the median estimate that should correspond to the 'loc' key
        if isinstance(y_pred, dict):
            if "loc" not in y_pred:
                raise ValueError("y_pred should contain a 'loc' key when it is a dict.")

            y_pred = y_pred["loc"]

        return np.abs(y_true - y_pred)


class NormalNegLogLikelihood(Loss):
    """The negative log-likelihood of a normal distribution.

    Given observed data ``y_true`` and the predicted parameters of the normal distribution
    ``y_pred["loc"], y_pred["scale"]``.
    """

    def __init__(self):
        self.dist = ss.norm

    def __call__(
        self, y_true: np.ndarray, y_pred: np.ndarray | Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Compute the negative log-likelihood of a normal distribution.

        Args:
            y_true (np.ndarray): the true target values.
            y_pred (np.ndarray or Dict[str, np.ndarray]): the predicted target values. If it is a
                _Dict[str, np.ndarray]_ then it should contain a key ``"loc"``.

        Returns:
            np.ndarray: the loss value with first dimension ``n_samples``.
        """
        _check_is_array(y_true, "y_true")
        _check_is_array_or_dict(y_pred, "y_pred")

        if isinstance(y_pred, np.ndarray):
            if np.shape(y_pred)[0] != 2:
                raise ValueError(
                    "The first dimension of y_pred should be equal to 2 but it is "
                    f"{np.shape(y_pred)[0]}."
                )

            if np.shape(y_pred)[1:] != np.shape(y_true):
                raise ValueError(
                    f"{np.shape(y_true)=} and {np.shape(y_pred)=} do not have matching shapes."
                )

            y_pred_loc = y_pred[0]
            y_pred_scale = y_pred[1]

        else:
            if "loc" not in y_pred:
                raise ValueError("y_pred should contain a 'loc' key when it is a dict.")

            if "scale" not in y_pred:
                raise ValueError("y_pred should contain a 'scale' key when it is a dict.")

            y_pred_loc = y_pred["loc"]
            y_pred_scale = y_pred["scale"]

        return -self.dist.logpdf(y_true, loc=y_pred_loc, scale=y_pred_scale)


class ZeroOneLoss(Loss):
    """Zero-One loss for classification (a.k.a, error rate).

    It has value ``1`` if the prediction is wrong and ``0`` if it is correct.

    Args:
        predict_proba (bool, optional): A boolean indicating if ``y_pred`` contains predicted
            categorical probabilities. Defaults to ``False`` for label predictions.
    """

    def __init__(self, predict_proba: bool = False):
        self._predict_proba = predict_proba

    def __call__(
        self, y_true: np.ndarray, y_pred: np.ndarray | Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Compute the zero-one loss.

        Args:
            y_true (np.ndarray): the true target values. It should be an array of integers
                representing the true class labels.
            y_pred (np.ndarray or Dict[str, np.ndarray]): the predicted target values. If it is a
                _Dict[str, np.ndarray]_ then it should contain a key ``"loc"``.

        Returns:
            np.ndarray: the loss value with first dimension ``n_samples``.
        """
        _check_is_array(y_true, "y_true")
        _check_is_array_or_dict(y_pred, "y_pred")

        if isinstance(y_pred, dict):
            if "loc" not in y_pred:
                raise ValueError("y_pred should contain a 'loc' key when it is a dict.")

            y_pred = y_pred["loc"]

        if self._predict_proba:
            y_pred = np.argmax(y_pred, axis=-1)

            if len(np.shape(y_true)) == len(np.shape(y_pred)) + 1:
                y_pred = np.expand_dims(y_pred, -1)

        return np.array(y_true != y_pred, dtype=np.float64)


class CategoricalCrossEntropy(Loss):
    """Categorical-Cross Entropy (a.k.a., Log-Loss) function for classification."""

    def __call__(
        self, y_true: np.ndarray, y_pred: np.ndarray | Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Compute the categorical crossentropy loss.

        Args:
            y_true (np.ndarray): the true target values. It should be an array of labels or one-hot
                encoded labels representing the true class labels.
            y_pred (np.ndarray or Dict[str, np.ndarray]): the predicted target values. If it is a
                _Dict[str, np.ndarray]_ then it should contain a key ``"loc"``. It is an array of
                predicted categorical probabilities.

        Returns:
            np.ndarray: the loss value with first dimension ``n_samples``.
        """
        _check_is_array(y_true, "y_true")
        _check_is_array_or_dict(y_pred, "y_pred")

        if isinstance(y_pred, dict):
            if "loc" not in y_pred:
                raise ValueError("y_pred should contain a 'loc' key when it is a dict.")

            y_pred = y_pred["loc"]

        if len(np.shape(y_true)) == len(np.shape(y_pred)) - 1:
            # Then the passed y_true is an array of integers
            num_classes = np.shape(y_pred)[-1]
            y_true = np.eye(num_classes)[y_true]

        eps = np.finfo(y_pred.dtype).eps
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return (-np.log(y_pred) * y_true).sum(axis=-1)

"""Submodule of transformer (i.e., preprocessing pipelines) for Skopt."""

from __future__ import division

import numpy as np
from sklearn.preprocessing import LabelBinarizer


class Transformer(object):
    """Base class for all 1-D transformers."""

    def fit(self, X):
        """Initialize the transformer with current ``X``."""
        return self

    def transform(self, X):
        """Applies the transformation."""
        raise NotImplementedError

    def inverse_transform(self, X):
        """Reverse the transformation."""
        raise NotImplementedError


class Identity(Transformer):
    """Identity transform."""

    def __init__(self, type_func=None) -> None:
        self.type_func = type_func

    def transform(self, X):
        """Returns the identity."""
        return X

    def inverse_transform(self, Xt):
        """Returns the identity."""
        if self.type_func:
            return [self.type_func(Xt[0])]
        else:
            return Xt


class ToInteger(Transformer):
    """Identity transform."""

    def transform(self, X):
        """Apply the integer transform."""
        return int(X)

    def inverse_transform(self, Xt):
        """Returns the identity as the operation is not reversable."""
        return Xt


class StringEncoder(Transformer):
    """StringEncoder transform.

    The transform will cast everything to a string and the inverse transform will cast to the type defined in dtype.
    """

    def __init__(self, dtype=str):
        super(StringEncoder, self).__init__()
        self.dtype = dtype

    def fit(self, X):
        """Fit a list or array of categories. All elements must be from the same type.

        Args:
        X : array-like, shape=(n_categories,)
            List of categories.
        """
        if len(X) > 0:
            self.dtype = type(X[0])

    def transform(self, X):
        """Transform an array of categories to a string encoded representation.

        Args:
            X (array-like, shape=(n_samples,)): List of categories.

        Returns:
            Xt (array-like, shape=(n_samples,)): The string encoded categories.
        """
        return [str(x) for x in X]

    def inverse_transform(self, Xt):
        """Inverse transform string encoded categories back to their original representation.

        Args:
            Xt (array-like, shape=(n_samples,)): String encoded categories.

        Returns:
            X (array-like, shape=(n_samples,)): The original categories.
        """
        return [self.dtype(x) for x in Xt]


class LogN(Transformer):
    """Base N logarithm transform."""

    def __init__(self, base):
        self._base = base

    def transform(self, X):
        """Apply the log transform."""
        return np.log10(np.asarray(X, dtype=float)) / np.log10(self._base)

    def inverse_transform(self, Xt):
        """Reverse the log transform."""
        return self._base ** np.asarray(Xt, dtype=float)


class CategoricalEncoder(Transformer):
    """OneHotEncoder that can handle categorical variables."""

    def __init__(self):
        """Convert labeled categories into one-hot encoded features."""
        self._lb = LabelBinarizer()

    def fit(self, X):
        """Fit a list or array of categories.

        Args:
            X (array-like, shape=(n_categories,)): List of categories.
        """
        self.mapping_ = {v: i for i, v in enumerate(X)}
        self.inverse_mapping_ = {i: v for v, i in self.mapping_.items()}
        self._lb.fit([self.mapping_[v] for v in X])
        self.n_classes = len(self._lb.classes_)

        return self

    def transform(self, X):
        """Transform an array of categories to a one-hot encoded representation.

        Args:
            X (array-like, shape=(n_samples,)): List of categories.

        Returns:
            Xt (array-like, shape=(n_samples, n_categories)): The one-hot encoded categories.
        """
        return self._lb.transform([self.mapping_[v] for v in X])

    def inverse_transform(self, Xt):
        """Inverse transform one-hot encoded categories back to their original representation.

        Args:
            Xt (array-like, shape=(n_samples, n_categories)): One-hot encoded categories.

        Returns:
            X (array-like, shape=(n_samples,)): The original categories.
        """
        Xt = np.asarray(Xt)
        return [self.inverse_mapping_[i] for i in self._lb.inverse_transform(Xt)]


class LabelEncoder(Transformer):
    """LabelEncoder that can handle categorical variables."""

    def __init__(self, X=None):
        if X is not None:
            self.fit(X)

    def fit(self, X):
        """Fit a list or array of categories.

        Args:
            X (array-like, shape=(n_categories,)): List of categories.
        """
        X = np.asarray(X)
        if X.dtype == object:
            self.mapping_ = {v: i for i, v in enumerate(X)}
        else:
            i = 0
            self.mapping_ = {}
            for v in np.unique(X):
                self.mapping_[v] = i
                i += 1
        self.inverse_mapping_ = {i: v for v, i in self.mapping_.items()}
        return self

    def transform(self, X):
        """Transform an array of categories to a one-hot encoded representation.

        Args:
            X (array-like, shape=(n_samples,)): List of categories.

        Returns:
            Xt (array-like, shape=(n_samples, n_categories)): The integer categories.
        """
        X = np.asarray(X)
        return [self.mapping_[v] for v in X]

    def inverse_transform(self, Xt):
        """Inverse transform integer categories back to their original representation.

        Args:
            Xt (array-like, shape=(n_samples, n_categories)): Integer categories.

        Returns:
            X (array-like, shape=(n_samples,)): The original categories.
        """
        if isinstance(Xt, (float, np.float64)):
            Xt = [Xt]
        else:
            Xt = np.asarray(Xt)
        return [self.inverse_mapping_[int(np.round(i))] for i in Xt]


class Normalize(Transformer):
    """Scales each dimension into the interval [0, 1].

    Args:
        low (float): Lower bound.
        high (float): Higher bound.
        is_int (bool): Round and cast the return value of `inverse_transform` to integer. Set
            to `True` when applying this transform to integers. Defaults to ``False``.
    """

    def __init__(self, low, high, is_int=False):
        self.low = float(low)
        self.high = float(high)
        self.is_int = is_int
        self._eps = 1e-8

    def transform(self, X):
        """Apply the normalization."""
        X = np.asarray(X)
        if self.is_int:
            if np.any(np.round(X) > self.high):
                raise ValueError(
                    "All integer values should" "be less than %f" % self.high
                )
            if np.any(np.round(X) < self.low):
                raise ValueError(
                    "All integer values should" "be greater than %f" % self.low
                )
        else:
            if np.any(X > self.high + self._eps):
                raise ValueError("All values should" "be less than %f" % self.high)
            if np.any(X < self.low - self._eps):
                raise ValueError("All values should" "be greater than %f" % self.low)
        if (self.high - self.low) == 0.0:
            return X * 0.0
        if self.is_int:
            return (np.round(X).astype(int) - self.low) / (self.high - self.low)
        else:
            return (X - self.low) / (self.high - self.low)

    def inverse_transform(self, X):
        """Reverse the normalization."""
        X = np.asarray(X)
        if np.any(X > 1.0 + self._eps):
            raise ValueError("All values should be less than 1.0")
        if np.any(X < 0.0 - self._eps):
            raise ValueError("All values should be greater than 0.0")
        X_orig = X * (self.high - self.low) + self.low
        if self.is_int:
            return np.round(X_orig).astype(int)
        return X_orig


class Pipeline(Transformer):
    """A lightweight pipeline to chain transformers.

    Args:
        transformers (list): A list of Transformer instances.
    """

    def __init__(self, transformers):
        self.transformers = list(transformers)
        for transformer in self.transformers:
            if not isinstance(transformer, Transformer):
                raise ValueError(
                    "Provided transformers should be a Transformer "
                    "instance. Got %s" % transformer
                )

    def fit(self, X):
        """Initialize all transformers with current ``X``."""
        for transformer in self.transformers:
            transformer.fit(X)
        return self

    def transform(self, X):
        """Apply the preprocessing mapping for all elements of ``X``."""
        for transformer in self.transformers:
            X = transformer.transform(X)
        return X

    def inverse_transform(self, X):
        """Reverse the preprocessing mapping for all elements of ``X``."""
        for transformer in self.transformers[::-1]:
            X = transformer.inverse_transform(X)
        return X
